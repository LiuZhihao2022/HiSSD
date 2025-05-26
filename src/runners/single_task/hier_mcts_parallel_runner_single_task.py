import numpy as np
import torch as th
import torch.nn.functional as F
import copy
import sys
sys.path.append('/home/lzh/HiSSD') # Ensure HiSSD is in path
import logging
import pickle
import cloudpickle
import functools
import mctx
import os

# Setup matplotlib logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)
import matplotlib.pyplot as plt

from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe, Process

# Imports from policy_improvement_demo.py and recurrent_fn.py
from typing import Tuple, Optional
from absl import app
from absl import flags
from mctx._src.network import PolicyRNN, ReplayBuffer, compute_prior_from_qvalues
from mctx._src.optimizer_wrapper import ValueOptimizerWrapper
from mctx._src.simple_env import SimpleEnv
from mctx._src.recurrent_fn import make_recurrent_fn_gym, make_recurrent_fn_world_model, make_multiagent_recurrent_fn_gym, make_recurrent_fn_real_simulator
from mctx._src.utils import stochastic_top_k_sampling
from mctx._src import action_selection
from examples.policy_improvement_demo import initialize_root, DemoOutput


import jax
import jax.numpy as jnp
import pandas as pd

jax.config.update('jax_debug_nans', True)
logging.getLogger('jax').setLevel(logging.INFO)
logging.getLogger('absl').setLevel(logging.WARNING)


class HierMCTSParallelRunner:

    def __init__(self, args, logger, task): # task is just a name for single-task
        self.args = args
        self.logger = logger
        self.task = task # Name of the single task (e.g., map name)
        self.batch_size = self.args.batch_size_run

        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])
        env_fn = env_REGISTRY[self.args.env]

        worker_id2env_args = {}
        for worker_id in range(self.batch_size):
            worker_id2env_args[worker_id] = copy.deepcopy(self.args.env_args)
            # Ensure env_args has 'seed' if it's expected by the env_fn
            if "seed" not in worker_id2env_args[worker_id] and hasattr(self.args, "seed"):
                worker_id2env_args[worker_id]["seed"] = getattr(self.args, "seed", 0) + worker_id
            
            if "seed" in worker_id2env_args[worker_id]:
                worker_id2env_args[worker_id]["seed"] += worker_id
            else:
                worker_id2env_args[worker_id]["seed"] = worker_id

        self.ps = [Process(target=env_worker, 
                           args=(worker_conn, CloudpickleWrapper(partial(env_fn, **worker_id2env_args[worker_id])))) 
                   for worker_id, worker_conn in enumerate(self.worker_conns)]

        for p in self.ps:
            p.daemon = True
            p.start()

        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]

        self.t = 0
        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self.log_train_stats_t = -100000
        
        self.c_step = args.c_step
        self.num_simulations = getattr(args, "num_simulations", 32)
        self.max_num_considered_actions = getattr(args, "max_num_considered_actions", 16)
        self.use_mixed_value = getattr(args, "use_mixed_value", False)
        self.k = getattr(args, "k", 10) # k for Gumbel top-k
        self.temperature = getattr(args, "temperature", 1.0)

        self.current_skill_indices = [None for _ in range(self.batch_size)]
        self.current_mcts_data = [None for _ in range(self.batch_size)]
        self.last_skill_selection_t = [0 for _ in range(self.batch_size)]
        self.accumulated_rewards = [0 for _ in range(self.batch_size)]
        self.wm_hidden_states = None # World model hidden states
        
        # JAX PRNG key
        env_seed = self.args.env_args.get('seed', self.args.seed if hasattr(self.args, "seed") else 0)
        self.rng_key = jax.random.PRNGKey(env_seed)
        
        root_action_selection_fn=functools.partial(
          action_selection.gumbel_muzero_root_action_selection,
          num_simulations=self.num_simulations,
          max_num_considered_actions=self.max_num_considered_actions,
          qtransform=functools.partial(
                mctx.qtransform_completed_by_mix_value,
                use_mixed_value=self.use_mixed_value,
            ),
        )
        
        interior_action_selection_fn=functools.partial(
            action_selection.gumbel_muzero_interior_action_selection,
            qtransform=functools.partial(
                mctx.qtransform_completed_by_mix_value,
                use_mixed_value=self.use_mixed_value,
            ),
        )

        self.action_selection_fn = action_selection.switching_action_selection_wrapper(
            root_action_selection_fn=root_action_selection_fn,
            interior_action_selection_fn=interior_action_selection_fn
        )
        
        self.episode_count = 0
        self.distribution_record_interval = getattr(args, "distribution_record_interval", 16)
        self.distribution_records = []
        self.max_distributions_to_keep = getattr(args, "max_distributions_to_keep", 150)

        # For bandit test
        self.bandit_run_count = 0
        self.bandit_stats_history = {
            "mean_advantage": [],
            "mean_selected_action_value": [],
            "mean_prior_policy_action_value": [],
            "mean_action_weights_policy_value": [],
            "mean_root_value": [],
            "mean_q_value_advantage": [],
            "prior_action_probs": [],
            "prior_action_logits": [],
            "mcts_action_weights": []
        }


    def setup(self, scheme, groups, preprocess, mac, mcts_network):
        self.new_batch = partial(
            EpisodeBatch,
            scheme,
            groups,
            self.batch_size,
            self.episode_limit + 1,
            preprocess=preprocess,
            device=self.args.device,
        )
        self.mac = mac # Single-task MAC
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess
        self.mcts_network = mcts_network # MCTS policy/value network (PolicyRNN)
        
        self.n_agents = self.mac.n_agents # Access n_agents from single-task mac
        
        # Create recurrent function - choose between world model or real simulator
        use_real_simulator = getattr(self.args, "use_real_simulator", False)
        
        if use_real_simulator:
            # Create real simulator environment instance for MCTS
            env_fn = env_REGISTRY[self.args.env]
            env_args = copy.deepcopy(self.args.env_args)
            self.mcts_env = env_fn(**env_args)
            
            self.recurrent_fn = make_recurrent_fn_real_simulator(
                self.mac, 
                self.batch_size,
                self.mcts_network,
                self.temperature,
                self.n_agents,
                self.k,
                offline_value_start=self.args.offline_value_start,
                offline_value_end=self.args.offline_value_end,
                offline_value_anneal_time=self.args.offline_value_anneal_time,
                real_env=self.mcts_env
            )
        else:
            self.recurrent_fn = make_recurrent_fn_world_model(
                self.mac, 
                self.batch_size,
                self.mcts_network,
                self.temperature,
                self.n_agents,
                self.k,
                offline_value_start=self.args.offline_value_start,
                offline_value_end=self.args.offline_value_end,
                offline_value_anneal_time=self.args.offline_value_anneal_time
            )
        
        # Create MCTS replay buffers for each environment
        self.replay_buffers_mcts = [ReplayBuffer(
            self.episode_limit//self.c_step + 1,
            1,  # Each environment has one batch
            self.c_step,
            use_real_data=False
        ) for _ in range(self.batch_size)]


    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        pass

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))
        # Close MCTS environment if using real simulator
        if hasattr(self, 'mcts_env') and self.mcts_env is not None:
            self.mcts_env.close()

    def reset(self):
        self.batch = self.new_batch()
        # Reset all environments
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))
            
        pre_transition_data = {
            "state": [],
            "avail_actions": [],
            "avail_skills": [],
            "obs": []
        }
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])
            if getattr(self.args, "basic_action_as_skill", False):
                pre_transition_data["avail_skills"].append(data["avail_actions"])
            else:
                pre_transition_data["avail_skills"].append(np.ones((self.n_agents, self.args.skill_dim)))

        self.batch.update(pre_transition_data, ts=0)

        self.t = 0
        self.env_steps_this_run = 0
        self.wm_hidden_states = None
        # Reset parallel state variables
        self.current_skill_indices = [None for _ in range(self.batch_size)]
        self.current_mcts_data = [None for _ in range(self.batch_size)]
        self.last_skill_selection_t = [0 for _ in range(self.batch_size)]
        self.accumulated_rewards = [0 for _ in range(self.batch_size)]
        self.replay_buffers_mcts = [ReplayBuffer(
            self.episode_limit//self.c_step + 1,
            1,
            self.c_step,
            use_real_data=False
        ) for _ in range(self.batch_size)]

    def run(self, test_mode=False, nolog=False, pretrain=False):
        self.reset()

        episode_returns = [0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        self.mac.init_hidden(batch_size=self.batch_size) # Remove task parameter
        self.mcts_network.init_hidden(batch_size=self.batch_size)
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        final_env_infos = [None for _ in range(self.batch_size)]
        predict_reward = []
        true_reward = []
        
        # Record distribution data
        record_distribution = False
        if not test_mode and not pretrain and self.episode_count % self.distribution_record_interval == 0:
            record_distribution = True
        
        while True:
            # 1. Batch skill selection
            skill_select_envs = []
            for idx in envs_not_terminated:
                if episode_lengths[idx] % self.c_step == 0:
                    skill_select_envs.append(idx)
            if not pretrain and skill_select_envs:
                skill_indices, mcts_datas, new_wm_hidden_states, new_policy_hidden_states, new_critic_hidden_states, distribution_data = self._batch_select_skill_with_mcts(skill_select_envs, record_distribution=record_distribution)
                
                # Record distribution data if needed
                if record_distribution and distribution_data:
                    self.distribution_records.append(distribution_data)
                    if len(self.distribution_records) > self.max_distributions_to_keep:
                        self.distribution_records.pop(0)
                    self.plot_distribution()
                    record_distribution = False
                
                for i, env_idx in enumerate(skill_select_envs):
                    if self.current_mcts_data[env_idx] is not None and episode_lengths[env_idx] > 0:
                        current_input = self.mac._build_inputs(self.batch[env_idx], t=self.t, use_skill=True).reshape(1, self.n_agents, -1)
                        current_state = self.batch["state"][env_idx:env_idx+1, self.t]
                        policy_output, experienced_thresholds, advantages, root_policy_hidden_state, root_critic_hidden_state = self.current_mcts_data[env_idx]
                        pred_reward_for_action = policy_output.search_tree.children_rewards[0,0,policy_output.action.item()].item()
                        predict_reward.append(pred_reward_for_action)
                        true_reward.append(self.accumulated_rewards[env_idx])
                        self.replay_buffers_mcts[env_idx].push(
                            policy_output, 
                            experienced_thresholds, 
                            advantages, 
                            root_policy_hidden_state, 
                            root_critic_hidden_state,
                            real_r=self.accumulated_rewards[env_idx],
                            real_next_obs=current_input,
                            real_next_state=current_state,
                            real_done=np.zeros(1, dtype=bool)
                        )
                        self.accumulated_rewards[env_idx] = 0
                    self.current_skill_indices[env_idx] = skill_indices[i]
                    self.current_mcts_data[env_idx] = mcts_datas[i]
                    self.last_skill_selection_t[env_idx] = episode_lengths[env_idx]
                    self.mcts_network.policy_hidden = new_policy_hidden_states
                    self.mcts_network.critic_hidden = new_critic_hidden_states

            # 2. Batch action selection
            if pretrain:
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=0, bs=envs_not_terminated, test_mode=False)
            else:   
                skill_indices = self.current_skill_indices
                
                if getattr(self.args, "basic_action_as_skill", False):
                    actions = th.tensor(self.current_skill_indices, dtype=th.int64)[envs_not_terminated]
                else:
                    if not self.args.use_origin_model:
                        actions = self.mac.forward_action_skill(
                            self.batch,
                            t=self.t,
                            skill_index=skill_indices,
                            test_mode=True,
                        )
                        avail_actions = self.batch["avail_actions"][:, self.t]
                        actions = self.mac.action_selector.select_action(
                            actions[envs_not_terminated],
                            avail_actions[envs_not_terminated],
                            t_env=self.t_env,
                            test_mode=True,
                        )
            
            cpu_actions = actions.to("cpu").numpy()

            # 3. Update actions
            action_idx = actions.squeeze(1)
            skill_idx = th.tensor([self.current_skill_indices[i] for i in envs_not_terminated], device=self.args.device, dtype=th.long).squeeze()

            # One-hot encoding
            num_actions = self.batch["avail_actions"].shape[-1]
            actions_onehot = F.one_hot(action_idx, num_classes=num_actions).float().unsqueeze(1)
            skills_onehot  = F.one_hot(skill_idx.long(), num_classes=self.args.skill_dim).float().unsqueeze(1)

            actions_chosen = {
                "actions":        actions.unsqueeze(1),
                "actions_onehot": actions_onehot,
                "skills":         skill_idx.unsqueeze(1),
                "skills_onehot":  skills_onehot
            }
            
            self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # 4. Environment step
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated:
                    if not terminated[idx]:
                        parent_conn.send(("step", cpu_actions[action_idx]))
                    action_idx += 1
            
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            all_terminated = all(terminated)
            if all_terminated:
                break
                
            # 5. Collect environment feedback
            post_transition_data = {
                "reward": [],
                "terminated": [],
            }
            pre_transition_data = {
                "state": [],
                "avail_actions": [],
                "avail_skills": [],
                "obs": []
            }
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()
                    post_transition_data["reward"].append((data["reward"],))
                    episode_returns[idx] += data["reward"]
                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1
                    if not pretrain:
                        self.accumulated_rewards[idx] = self.args.gamma * self.accumulated_rewards[idx] + data["reward"]
                    env_terminated = False
                    if data["terminated"]:
                        final_env_infos[idx] = data["info"]
                        if not pretrain and self.current_mcts_data[idx] is not None:
                            final_input = self.mac._build_inputs(self.batch, t=self.t, use_skill=True).reshape(1, self.n_agents, -1)
                            final_state = data["state"][np.newaxis, :]
                            policy_output, experienced_thresholds, advantages, root_policy_hidden_state, root_critic_hidden_state = self.current_mcts_data[idx]
                            self.replay_buffers_mcts[idx].push(
                                policy_output, 
                                experienced_thresholds, 
                                advantages, 
                                root_policy_hidden_state, 
                                root_critic_hidden_state,
                                real_r=self.accumulated_rewards[idx],
                                real_next_obs=final_input,
                                real_next_state=final_state,
                                real_done=np.ones(1, dtype=bool)
                            )
                    if data["terminated"] and not data["info"].get("episode_limit", False):
                        env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))
                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["obs"].append(data["obs"])
                    if getattr(self.args, "basic_action_as_skill", False):
                        pre_transition_data["avail_skills"].append(data["avail_actions"])
                    else:
                        pre_transition_data["avail_skills"].append(np.ones((self.n_agents, self.args.skill_dim)))
                        
            # 6. Update batch
            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)
            self.t += 1
            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)

        if not test_mode:
            self.t_env += self.env_steps_this_run
            self.episode_count += self.batch_size
            
        replay_buffers = self.replay_buffers_mcts if not pretrain else None

        # Record statistics
        result_info = {}
        if not pretrain:
            cur_stats = self.test_stats if test_mode else self.train_stats
            cur_returns = self.test_returns if test_mode else self.train_returns
            log_prefix = f"test_" if test_mode else ""
            
            infos = [cur_stats] + [info for info in final_env_infos if info is not None]
            all_keys = set()
            for d in infos:
                all_keys |= set(d)
            
            # Handle different types of statistics when merging
            merged_stats = {}
            for k in all_keys:
                values = [d.get(k, 0) for d in infos if k in d]
                if not values:
                    merged_stats[k] = 0
                elif isinstance(values[0], (list, tuple)):
                    # For list/tuple fields like individual_rewards, concatenate them
                    merged_list = []
                    for v in values:
                        if isinstance(v, (list, tuple)):
                            merged_list.extend(v)
                        else:
                            merged_list.append(v)
                    merged_stats[k] = merged_list
                else:
                    # For scalar fields, sum them
                    merged_stats[k] = sum(values)
            
            merged_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
            merged_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)
            cur_stats.update(merged_stats)
            cur_returns.extend(episode_returns)
            
            n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
            if test_mode and (len(self.test_returns) == n_test_runs):
                self._log(cur_returns, cur_stats, log_prefix)
            elif not test_mode and self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
                self._log(cur_returns, cur_stats, log_prefix)
                if hasattr(self.mac.action_selector, "epsilon"):
                    self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
                self.log_train_stats_t = self.t_env

            avg_return = float(np.mean(episode_returns)) if episode_returns else 0.0
            avg_length = float(np.mean(episode_lengths)) if episode_lengths else 0.0
            
            # Handle environments that may not have battle_won field
            wins = []
            for info in final_env_infos:
                if info is not None and "battle_won" in info:
                    wins.append(info["battle_won"])
            avg_win = float(sum(wins)) / len(wins) if wins else 0.0

            stats_info = {}
            keys = set(k for info in final_env_infos if info for k in info)
            for k in keys:
                vals = [info.get(k, 0) for info in final_env_infos if info is not None and k in info]
                if vals:
                    if isinstance(vals[0], (list, tuple)):
                        # For list/tuple fields, don't average but keep as is or concatenate
                        stats_info[k] = vals[0] if len(vals) == 1 else [item for sublist in vals for item in (sublist if isinstance(sublist, (list, tuple)) else [sublist])]
                    else:
                        # For scalar fields, compute average
                        stats_info[k] = float(sum(vals) / len(vals))
                else:
                    stats_info[k] = 0.0

            result_info = {
                "episode_return": avg_return,
                "win_rate": avg_win,
                "episode_length": avg_length,
                # "stats": stats_info
            }
        else:
            result_info = {}
            
        predict_reward = np.array(predict_reward)
        true_reward = np.array(true_reward)
        mse = np.mean((predict_reward - true_reward) ** 2)
        print(f"MSE: {mse}")
        
        return self.batch, replay_buffers, result_info

    def _batch_select_skill_with_mcts(self, env_indices, record_distribution=False):
        batch_size = self.batch_size
        state_inputs = []
        obs_inputs = []
        
        for idx in range(self.batch_size):
            state = self.batch["state"][idx:idx+1, self.t].reshape(1, -1).cpu().numpy()
            obs = self.mac.preprocess_obs(self.batch[idx], self.t, use_skill=True).cpu().numpy() # Remove task parameter
            state_inputs.append(state)
            obs_inputs.append(obs)
            
        if self.wm_hidden_states is None:
            hidden_states_reward = self.mac.hidden_states_reward.clone()
            hidden_states_value = self.mac.hidden_states_value.clone()
            hidden_states_reward = hidden_states_reward.unsqueeze(1)
            hidden_states_value = hidden_states_value.unsqueeze(1)
            wm_hidden_states = th.cat([hidden_states_reward, hidden_states_value], dim=1)
            self.wm_hidden_states = wm_hidden_states.clone()
        else:
            self.wm_hidden_states[:,1] = self.mac.hidden_states_value.reshape(self.batch_size, self.n_agents, -1)
            wm_hidden_states = self.wm_hidden_states.clone()
            
        state_inputs = np.concatenate(state_inputs, axis=0)
        obs_inputs = np.concatenate(obs_inputs, axis=0)
        
        # Use mac for value prediction
        value_pre = self.mac.forward_value(self.batch, t=self.t) # Remove task parameter
        value_pre.unsqueeze_(1)
        state_input = self.batch["state"][:, self.t]
        state_input.unsqueeze_(1)
        outer_value = self.mac.mixer(value_pre, state_input) # Remove task_decomposer parameter
        
        # Batch initialize root
        roots, experienced_thresholds, root_policy_hidden_states, root_critic_hidden_states = initialize_root(
            self.mcts_network,
            state_inputs,
            obs_inputs,
            self.k,
            self.n_agents,
            self.args.skill_dim,
            wm_hidden_states=wm_hidden_states,
            bs_id=list(range(self.batch_size)),
            outer_value=outer_value,
            avail_skills=self.batch["avail_actions"][:, self.t] if getattr(self.args, "basic_action_as_skill", False) else None,
        )
        
        # Batch run MCTS
        policy_output, timing_stats, advantages, new_wm_hidden_states, new_policy_hidden_states, new_critic_hidden_states, rng_key = mctx.gumbel_muzero_policy(
            params=(),
            rng_key=self.rng_key,
            root=roots,
            recurrent_fn=self.recurrent_fn,
            action_selection_fn=self.action_selection_fn,
            num_simulations=self.num_simulations,
            task=None,
            current_t_env=self.t_env,
            args=self.args,
            max_num_considered_actions=self.max_num_considered_actions,
            qtransform=functools.partial(
                mctx.qtransform_completed_by_mix_value,
                use_mixed_value=self.use_mixed_value,
            ),
        )
        
        self.rng_key = rng_key
        skill_indices = []
        mcts_datas = []

        # Update hidden states
        new_wm_hidden_states = th.tensor(np.array(new_wm_hidden_states)).to(self.args.device)
        self.wm_hidden_states = new_wm_hidden_states
        new_policy_hidden_states = th.tensor(np.array(new_policy_hidden_states)).to(self.args.device)
        new_critic_hidden_states = th.tensor(np.array(new_critic_hidden_states)).to(self.args.device)
        
        for i, idx in enumerate(env_indices):
            skill_indices.append(policy_output[idx].chosen_skill)
            mcts_datas.append((
                policy_output[idx],
                experienced_thresholds[idx:idx+1],
                advantages[idx:idx+1],
                root_policy_hidden_states[idx:idx+1],
                root_critic_hidden_states[idx:idx+1]
            ))
        
        # Record distribution data
        distribution_data = None
        if record_distribution:
            b_idx = env_indices[0] if env_indices else 0
            batch_range = jnp.array([b_idx])
            root_idx = 0
            
            prior_logits = policy_output.search_tree.children_prior_logits[b_idx, root_idx]
            prior_probs = jax.nn.softmax(prior_logits)
            action_weights = policy_output.action_weights[b_idx]
            qvalues = policy_output.search_tree.qvalues(batch_range)[0, root_idx]
            root_value = roots.value[b_idx]
            
            obs = self.batch["obs"][b_idx, self.t].cpu().numpy()
            state = self.batch["state"][b_idx, self.t].cpu().numpy()
            
            distribution_data = {
                "episode": self.episode_count,
                "env_idx": b_idx,
                "t": self.t_env,
                "prior_logits": np.array(prior_logits),
                "prior_probs": np.array(prior_probs),
                "mcts_action_weights": np.array(action_weights),
                "qvalues": np.array(qvalues),
                "root_value": float(root_value),
                "obs_summary": f"obs-norm: {np.linalg.norm(obs)}",
                "state_summary": f"state-norm: {np.linalg.norm(state)}",
                "timestamp": np.datetime64('now')
            }
        
        skill_indices = np.array(skill_indices).reshape(len(env_indices), -1)
        return skill_indices, mcts_datas, new_wm_hidden_states, new_policy_hidden_states, new_critic_hidden_states, distribution_data

    def plot_distribution(self, save_dir="./results/mcts_distributions"):
        if not self.distribution_records:
            print("No distribution data available for plotting")
            return
        
        os.makedirs(save_dir, exist_ok=True)
        
        record = self.distribution_records[-1]
        
        prior_logits = record["prior_logits"]
        prior_probs = record["prior_probs"]
        mcts_weights = record["mcts_action_weights"]
        qvalues = record["qvalues"]
        root_value = record["root_value"]
        episode = record["episode"]
        env_idx = record["env_idx"]
        t = record["t"]
        
        x = np.arange(len(prior_probs))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(14, 10))
        rects1 = ax.bar(x - width/2, prior_probs, width, label='Prior Probability')
        rects2 = ax.bar(x + width/2, mcts_weights, width, label='MCTS After Search')
        
        ax.set_xlabel('Action Index', fontsize=12)
        ax.set_ylabel('Probability', fontsize=12)
        ax.set_title(f'Episode {episode}, Env {env_idx}, T {t}: Prior vs MCTS Probability Comparison', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        eps = 1e-10
        kl_div = np.sum(prior_probs * np.log((prior_probs + eps) / (mcts_weights + eps)))
        prior_entropy = -np.sum(prior_probs * np.log2(prior_probs + eps))
        mcts_entropy = -np.sum(mcts_weights * np.log2(mcts_weights + eps))
        
        ax.text(0.02, 0.95, 
                f"KL Divergence: {kl_div:.4f}\n"
                f"Prior Entropy: {prior_entropy:.4f}\n"
                f"MCTS Entropy: {mcts_entropy:.4f}\n"
                f"Root Value: {root_value:.4f}", 
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        threshold = 0.01
        for i, (prior, mcts, logit) in enumerate(zip(prior_probs, mcts_weights, prior_logits)):
            if prior > threshold:
                ax.text(i - width/2, prior/2, f'{logit:.2f}', 
                        ha='center', va='center', fontsize=8, color='white', 
                        fontweight='bold', rotation=90)
                ax.text(i - width/2, prior, f'{prior:.2f}', 
                        ha='center', va='bottom', fontsize=8)
            if mcts > threshold:
                ax.text(i + width/2, mcts, f'{mcts:.2f}', 
                        ha='center', va='bottom', fontsize=8)
        
        compare_path = os.path.join(save_dir, f"action_probs_compare_ep{episode}_env{env_idx}_t{t}.png")
        plt.savefig(compare_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"Prior vs MCTS probability comparison chart saved to {compare_path}")
        
        # ...existing code for other plotting functionality...

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                # Only compute mean for numeric values, skip list/tuple types
                if isinstance(v, (list, tuple)):
                    # For list/tuple data, we could log length or other summary stats
                    if len(v) > 0 and isinstance(v[0], (int, float)):
                        # If it's a list of numbers, compute mean of the list items
                        self.logger.log_stat(prefix + k + "_mean", np.mean(v), self.t_env)
                    # For non-numeric lists, skip logging or log length
                    continue
                elif isinstance(v, (int, float)) and stats["n_episodes"] > 0:
                    # For scalar numeric values, compute average across episodes
                    self.logger.log_stat(prefix + k + "_mean", v/stats["n_episodes"], self.t_env)
        stats.clear()

    def save_initial_state(self):
        self.reset()
        
        self.bandit_batch = copy.deepcopy(self.batch)
        self.bandit_t = 0
        
        self.mac.init_hidden(batch_size=self.batch_size) # Remove task parameter
        self.mcts_network.init_hidden(batch_size=self.batch_size)
        
        self.bandit_policy_hidden = self.mcts_network.policy_hidden.clone()
        self.bandit_critic_hidden = self.mcts_network.critic_hidden.clone()
        
        if hasattr(self.mac, 'hidden_states_reward') and hasattr(self.mac, 'hidden_states_value'):
            hidden_states_reward = self.mac.hidden_states_reward.clone()
            hidden_states_value = self.mac.hidden_states_value.clone()
            hidden_states_reward = hidden_states_reward.unsqueeze(1)
            hidden_states_value = hidden_states_value.unsqueeze(1)
            self.bandit_wm_hidden = th.cat([hidden_states_reward, hidden_states_value], dim=1)
        else:
            self.bandit_wm_hidden = None
            
        print("初始状态已保存，可以使用test_mcts_bandit进行固定状态测试")

    def test_mcts_bandit(self, num_trials=1):
        if not hasattr(self, 'bandit_batch'):
            print("请先调用save_initial_state()保存初始状态")
            return None, None, {}
        
        bandit_replay_buffer = ReplayBuffer(
            num_trials,
            self.batch_size,
            self.c_step,
            use_real_data=False
        )
        
        prior_actions = []
        mcts_actions = []
        advantages = []
        qvalues_data = []
        root_values = []
        q_value_advantages = []
        prior_probs_list = []
        prior_logits_list = []
        mcts_action_weights_list = []
        
        bandit_stats = {
            "mean_advantage": 0.0,
            "mean_selected_action_value": 0.0,
            "mean_prior_policy_action_value": 0.0,
            "action_weight_policy_value": 0.0,
        }
        
        for trial in range(num_trials):
            batch = copy.deepcopy(self.bandit_batch)
            
            self.mcts_network.policy_hidden = self.bandit_policy_hidden.clone()
            self.mcts_network.critic_hidden = self.bandit_critic_hidden.clone()
            
            if self.bandit_wm_hidden is not None:
                if hasattr(self.mac, 'hidden_states_reward') and hasattr(self.mac, 'hidden_states_value'):
                    self.mac.hidden_states_reward = self.bandit_wm_hidden[:, 0, :].clone()
                    self.mac.hidden_states_value = self.bandit_wm_hidden[:, 1, :].clone()
                self.wm_hidden_states = self.bandit_wm_hidden.clone()
                
            state_inputs = []
            obs_inputs = []
            
            for idx in range(self.batch_size):
                state = batch["state"][idx:idx+1, self.bandit_t].reshape(1, -1).cpu().numpy()
                obs = self.mac.preprocess_obs(batch[idx], self.bandit_t, use_skill=True).cpu().numpy() # Remove task parameter
                state_inputs.append(state)
                obs_inputs.append(obs)
                
            state_inputs = np.concatenate(state_inputs, axis=0)
            obs_inputs = np.concatenate(obs_inputs, axis=0)
            
            value_pre = self.mac.forward_value(batch, t=self.bandit_t) # Remove task parameter
            value_pre.unsqueeze_(1)
            state_input = batch["state"][:, self.bandit_t]
            state_input.unsqueeze_(1)
            outer_value = self.mac.mixer(value_pre, state_input) # Remove task_decomposer parameter
            
            roots, experienced_thresholds, root_policy_hidden_states, root_critic_hidden_states = initialize_root(
                self.mcts_network,
                state_inputs,
                obs_inputs,
                self.k,
                self.n_agents,
                self.args.skill_dim,
                wm_hidden_states=self.wm_hidden_states,
                bs_id=list(range(self.batch_size)),
                outer_value=outer_value,
                avail_skills=batch["avail_actions"][:, self.bandit_t] if getattr(self.args, "basic_action_as_skill", False) else None,
            )
            
            policy_output, timing_stats, advantages_data, new_wm_hidden_states, new_policy_hidden_states, new_critic_hidden_states, rng_key = mctx.gumbel_muzero_policy(
                params=(),
                rng_key=self.rng_key,
                root=roots,
                recurrent_fn=self.recurrent_fn,
                action_selection_fn=self.action_selection_fn,
                num_simulations=self.num_simulations,
                current_t_env=self.t_env,
                args=self.args,
                max_num_considered_actions=self.max_num_considered_actions,
                qtransform=functools.partial(
                    mctx.qtransform_completed_by_mix_value,
                    use_mixed_value=self.use_mixed_value,
                ),
            )
            
            self.rng_key = rng_key
            batch_range = jnp.arange(self.batch_size)
            root_idx = 0
            
            qvalues = policy_output.search_tree.qvalues(batch_range)
            prior_logits = policy_output.search_tree.children_prior_logits[batch_range, root_idx]
            
            prior_logits_list.append(np.array(prior_logits))
            
            prior_probs = jax.nn.softmax(prior_logits, axis=-1)
            prior_probs_list.append(np.array(prior_probs))
            
            selected_action = policy_output.action
            selected_action_value = qvalues[batch_range, root_idx, selected_action]
            
            gumbel = policy_output.search_tree.extra_data.root_gumbel
            prior_policy_action = jnp.argmax(gumbel + prior_logits, axis=-1)
            
            root_value = roots.value
            prior_policy_action_value = qvalues[batch_range, root_idx, prior_policy_action]
            
            q_value_advantage = selected_action_value - root_value
            
            print("先验动作: {}, 选择动作: {}".format(prior_policy_action, selected_action))
            print("根节点value: {:.4f}, 选择的Q值: {:.4f}, Q-V优势: {:.4f}".format(
                jnp.mean(root_value).item(), 
                jnp.mean(selected_action_value).item(),
                jnp.mean(q_value_advantage).item()
            ))
            
            action_weights = policy_output.action_weights
            mcts_action_weights_list.append(np.array(action_weights))
            action_weights_policy_value = jnp.sum(action_weights * qvalues[batch_range, root_idx], axis=-1)
            
            prior_actions.append(prior_policy_action)
            mcts_actions.append(selected_action)
            advantages.append(selected_action_value - prior_policy_action_value)
            root_values.append(root_value)
            q_value_advantages.append(q_value_advantage)
            
            qvalues_data.append({
                "selected_action": selected_action,
                "selected_action_value": selected_action_value,
                "prior_action": prior_policy_action,
                "prior_action_value": prior_action_value,
                "action_weights_policy_value": action_weights_policy_value,
                "root_value": root_value,
                "q_value_advantage": q_value_advantage
            })
            
            for b_idx in range(self.batch_size):
                bandit_replay_buffer.push(
                    policy_output[b_idx],
                    experienced_thresholds[b_idx:b_idx+1],
                    advantages_data[b_idx:b_idx+1],
                    root_policy_hidden_states[b_idx:b_idx+1],
                    root_critic_hidden_states[b_idx:b_idx+1],
                )
        
        bandit_stats = {
            "mean_advantage": np.mean(advantages),
            "mean_selected_action_value": np.mean([q["selected_action_value"] for q in qvalues_data]),
            "mean_prior_policy_action_value": np.mean([q["prior_action_value"] for q in qvalues_data]),
            "mean_action_weights_policy_value": np.mean([q["action_weights_policy_value"] for q in qvalues_data]),
            "mean_root_value": np.mean([q["root_value"] for q in qvalues_data]),
            "mean_q_value_advantage": np.mean([q["q_value_advantage"] for q in qvalues_data]),
            "prior_action_probs": np.mean(np.array(prior_probs_list), axis=0).astype(np.float64),
            "prior_action_logits": np.mean(np.array(prior_logits_list), axis=0).astype(np.float64),
            "mcts_action_weights": np.mean(np.array(mcts_action_weights_list), axis=0).astype(np.float64)
        }
        
        print(f"MCTS测试结果 ({num_trials} 次尝试):")
        print(f"平均优势: {bandit_stats['mean_advantage']:.4f}")
        print(f"MCTS选择动作的平均值: {bandit_stats['mean_selected_action_value']:.4f}")
        print(f"先验策略动作的平均值: {bandit_stats['mean_prior_policy_action_value']:.4f}")
        print(f"带权重的策略值平均: {bandit_stats['mean_action_weights_policy_value']:.4f}")
        print(f"根节点value平均值: {bandit_stats['mean_root_value']:.4f}")
        print(f"Q值相对于value的优势: {bandit_stats['mean_q_value_advantage']:.4f}")

        self.bandit_run_count += 1
        for key in self.bandit_stats_history:
            if key in bandit_stats:
                self.bandit_stats_history[key].append(bandit_stats[key])
        
        self.plot_bandit_stats()
        
        return batch, bandit_replay_buffer, bandit_stats

    def plot_bandit_stats(self, save_dir="./results/mcts_bandit"):
        os.makedirs(save_dir, exist_ok=True)
        
        if not self.bandit_stats_history["mean_advantage"]:
            print("没有可供绘制的历史数据")
            return
        
        # Plot scalar stats trends
        scalar_stats_to_plot = [
            "mean_advantage", "mean_selected_action_value", "mean_prior_policy_action_value",
            "mean_action_weights_policy_value", "mean_root_value", "mean_q_value_advantage"
        ]
        num_total_searches = self.bandit_run_count 
        if num_total_searches > 0 :
            num_points_per_stat = len(self.bandit_stats_history["mean_advantage"]) # Should be num_total_searches

            if num_points_per_stat > 0:
                fig_scalar, axes_scalar = plt.subplots(len(scalar_stats_to_plot), 1, figsize=(12, 4 * len(scalar_stats_to_plot)), sharex=True)
                if len(scalar_stats_to_plot) == 1: axes_scalar = [axes_scalar] # Ensure iterable

                x_axis_scalar = np.arange(num_points_per_stat) # Index of MCTS search instance

                for i, stat_name in enumerate(scalar_stats_to_plot):
                    axes_scalar[i].plot(x_axis_scalar, self.bandit_stats_history[stat_name], marker='o', linestyle='-', markersize=4)
                    axes_scalar[i].set_ylabel(stat_name, fontsize=10)
                    axes_scalar[i].grid(True, linestyle='--', alpha=0.7)
                axes_scalar[-1].set_xlabel("MCTS Search Instance Index", fontsize=12)
                fig_scalar.suptitle(f'MCTS Bandit Scalar Stats Trends (Total Searches: {num_total_searches})', fontsize=14)
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                scalar_fig_path = os.path.join(save_dir, f"bandit_scalar_trends_run{self.bandit_run_count}.png")
                plt.savefig(scalar_fig_path, dpi=150)
                plt.close(fig_scalar)
                self.logger.console_logger.info(f"Saved MCTS bandit scalar trends to {scalar_fig_path}")

        # Plot distribution comparison for the last recorded set of distributions
        # This requires taking the last self.batch_size items for each distribution type
        if self.bandit_stats_history["prior_action_probs"] and len(self.bandit_stats_history["prior_action_probs"]) >= self.batch_size:
            last_batch_prior_probs = np.mean(np.array(self.bandit_stats_history["prior_action_probs"][-self.batch_size:]), axis=0)
            last_batch_mcts_weights = np.mean(np.array(self.bandit_stats_history["mcts_action_weights"][-self.batch_size:]), axis=0)
            last_batch_prior_logits = np.mean(np.array(self.bandit_stats_history["prior_action_logits"][-self.batch_size:]), axis=0)

            fig_dist, ax_dist = plt.subplots(figsize=(15, 8))
            x_dist = np.arange(len(last_batch_prior_probs))
            bar_width_dist = 0.35

            ax_dist.bar(x_dist - bar_width_dist/2, last_batch_prior_probs, bar_width_dist, label='Avg Prior Probs (Last Batch)', color='deepskyblue')
            ax_dist.bar(x_dist + bar_width_dist/2, last_batch_mcts_weights, bar_width_dist, label='Avg MCTS Weights (Last Batch)', color='tomato')
            
            ax_dist.set_xlabel('Skill Index', fontsize=12)
            ax_dist.set_ylabel('Probability / Weight', fontsize=12)
            ax_dist.set_title(f'Avg Skill Distribution Comparison (Last Batch from Run {self.bandit_run_count})', fontsize=14)
            ax_dist.set_xticks(x_dist)
            ax_dist.legend(fontsize=10)
            ax_dist.grid(True, linestyle='--', alpha=0.7)

            for i in x_dist:
                ax_dist.text(x_dist[i] - bar_width_dist/2, last_batch_prior_probs[i] + 0.01, f'{last_batch_prior_logits[i]:.2f}', ha='center', va='bottom', fontsize=8, color='navy', rotation=90)
            
            dist_fig_path = os.path.join(save_dir, f"bandit_dist_compare_run{self.bandit_run_count}.png")
            plt.tight_layout()
            plt.savefig(dist_fig_path, dpi=150)
            plt.close(fig_dist)
            self.logger.console_logger.info(f"Saved MCTS bandit distribution comparison to {dist_fig_path}")


# Environment worker process
def env_worker(remote, env_fn):
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            reward, terminated, env_info = env.step(actions)
            state = env.get_state()
            avail_actions = env.get_avail_actions()
            obs = env.get_obs()
            remote.send({
                "state": state,
                "avail_actions": avail_actions,
                "obs": obs,
                "reward": reward,
                "terminated": terminated,
                "info": env_info
            })
        elif cmd == "reset":
            env.reset()
            state = env.get_state()
            remote.send({
                "state": state,
                "avail_actions": env.get_avail_actions(),
                "obs": env.get_obs()
            })
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info())
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        else:
            raise NotImplementedError


class CloudpickleWrapper():
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        self.x = pickle.loads(ob)

