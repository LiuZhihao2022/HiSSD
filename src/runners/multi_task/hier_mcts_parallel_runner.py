import numpy as np
import torch as th
import copy
import sys
sys.path.append('/home/lzh/HiSSD')
import logging
import pickle
import cloudpickle
import functools
import mctx

from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe, Process

# FROM policy_improvement_demo.py
sys.path.append('/home/lzh/HiSSD')
from typing import Tuple, Optional
from absl import app
from absl import flags
from mctx._src.network import PolicyRNN, ReplayBuffer, compute_prior_from_qvalues
from mctx._src.optimizer_wrapper import ValueOptimizerWrapper
from mctx._src.simple_env import SimpleEnv
from mctx._src.recurrent_fn import make_recurrent_fn_gym, make_recurrent_fn_world_model, make_multiagent_recurrent_fn_gym
from mctx._src.utils import stochastic_top_k_sampling
from mctx._src import action_selection
from examples.policy_improvement_demo import initialize_root, DemoOutput
import jax
jax.config.update('jax_debug_nans', True)
# 配置日志级别，减少调试信息
logging.getLogger('jax').setLevel(logging.INFO)
logging.getLogger('absl').setLevel(logging.WARNING)


class HierMCTSParallelRunner:

    def __init__(self, args, logger, task):
        self.args = args
        self.logger = logger
        self.task = task
        self.batch_size = self.args.batch_size_run

        # 创建环境子进程
        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])
        env_fn = env_REGISTRY[self.args.env]
        worker_id2env_args = {}
        for worker_id in range(self.batch_size):
            worker_id2env_args[worker_id] = copy.deepcopy(self.args.env_args)
            worker_id2env_args[worker_id]["seed"] += worker_id
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
        
        # MCTS相关参数
        self.c_step = args.c_step
        self.num_simulations = getattr(args, "num_simulations", 32)
        self.max_num_considered_actions = getattr(args, "max_num_considered_actions", 16)
        self.use_mixed_value = getattr(args, "use_mixed_value", False)
        self.k = getattr(args, "k", 10)
        self.temperature = getattr(args, "temperature", 1.0)

        # 新增：并行相关的状态变量
        self.current_skill_indices = [None for _ in range(self.batch_size)]
        self.current_mcts_data = [None for _ in range(self.batch_size)]
        self.last_skill_selection_t = [0 for _ in range(self.batch_size)]
        self.accumulated_rewards = [0 for _ in range(self.batch_size)]
        self.terminated = [False for _ in range(self.batch_size)]
        self.wm_hidden_states = None
        self.rng_key=jax.random.PRNGKey(self.args.env_args['seed']) 
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
        self.mac = mac
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess
        self.mcts_network = mcts_network
        
        # 获取任务相关信息
        self.n_agents = self.mac.task2n_agents[self.task]
        task_decomposer = self.mac.task2decomposer[self.task]
        self.n_enemy = task_decomposer.n_enemies
        self.n_ally = self.n_agents - 1
        
        # 为每个环境实例初始化状态
        self.wm_hidden_states = self.mac.init_hidden_wm(batch_size=self.batch_size, task=self.task)
        
        # 创建递归函数
        self.recurrent_fn = make_recurrent_fn_world_model(
            self.mac, 
            self.batch_size,  # 每个环境实例单独处理，所以batch_size=1
            self.mcts_network,
            self.temperature,
            self.n_agents,
            self.k
        )
        
        # 为每个环境创建MCTS回放缓冲区
        self.replay_buffers_mcts = [ReplayBuffer(
            self.episode_limit//self.c_step + 1,
            1,  # 每个环境一个batch
            self.c_step,
            use_real_data=True
        ) for _ in range(self.batch_size)]

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        pass

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self):
        self.batch = self.new_batch()

        # 重置所有环境
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))

        pre_transition_data = {
            "state": [],
            "avail_actions": [],
            "avail_skills": [],
            "obs": []
        }
        
        # 获取初始状态和观测
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])
            pre_transition_data["avail_skills"].append(np.ones((self.n_agents, self.args.skill_dim)))

        self.batch.update(pre_transition_data, ts=0)

        self.t = 0
        self.env_steps_this_run = 0
        
        # 并行相关状态变量初始化
        self.current_skill_indices = [None for _ in range(self.batch_size)]
        self.current_mcts_data = [None for _ in range(self.batch_size)]
        self.last_skill_selection_t = [0 for _ in range(self.batch_size)]
        self.accumulated_rewards = [0 for _ in range(self.batch_size)]
        self.terminated = [False for _ in range(self.batch_size)]
        self.wm_hidden_states = self.mac.init_hidden_wm(batch_size=self.batch_size, task=self.task)
        self.replay_buffers_mcts = [ReplayBuffer(
            self.episode_limit//self.c_step + 1,
            1,
            self.c_step,
            use_real_data=True
        ) for _ in range(self.batch_size)]

    def run(self, test_mode=False, nolog=False, pretrain=False):
        self.reset()

        episode_returns = [0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        self.mac.init_hidden(batch_size=self.batch_size, task=self.task)
        terminated = self.terminated
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        # 用于存储每个环境的最终env_info
        final_env_infos = [None for _ in range(self.batch_size)]

        while True:
            # 1. 批量skill选择
            skill_select_envs = []
            for idx in envs_not_terminated:
                if episode_lengths[idx] % self.c_step == 0:
                    skill_select_envs.append(idx)
            if not pretrain and skill_select_envs:
                skill_indices, mcts_datas, new_wm_hidden_states, new_policy_hidden_states, new_critic_hidden_states = self._batch_select_skill_with_mcts(skill_select_envs)
                for i, env_idx in enumerate(skill_select_envs):
                    if self.current_mcts_data[env_idx] is not None and episode_lengths[env_idx] > 0:
                        current_input = self.mac._build_inputs(self.batch[env_idx], t=self.t, task=self.task, use_skill=True).reshape(1, self.n_agents, -1)
                        current_state = self.batch["state"][env_idx:env_idx+1, self.t]
                        policy_output, experienced_thresholds, advantages, root_policy_hidden_state, root_critic_hidden_state = self.current_mcts_data[env_idx]
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
                    # 全部赋值影响不大。因为各个不同的batch之间的hidden state都是相互不影响的
                    self.wm_hidden_states = new_wm_hidden_states
                    self.mcts_network.policy_hidden = new_policy_hidden_states
                    self.mcts_network.critic_hidden = new_critic_hidden_states

            # 2. 批量动作选择
            if pretrain:
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=0, task=self.task, bs=envs_not_terminated, test_mode=False)
            else:   
                # 已经存在占位了，不会报错
                skill_indices = [self.current_skill_indices[idx] for idx in envs_not_terminated]
                # 这里一定要讲bs=envs_not_terminated传入，因为设计选取hidden state
                # 对于q mode（目前的mode），test_mode参数无影响，输出的是q值。如果是pi_logit mode，那么就没有随机性。
                actions = self.mac.forward_action_skill(
                    self.batch,
                    t=self.t,
                    skill_index=skill_indices,
                    task=self.task,
                    bs=envs_not_terminated,
                    test_mode=True,
                )
                # 选择基本动作时，不需要随机性，直接将skill解码
                actions = self.mac.action_selector.select_action(
                    actions,
                    self.batch[envs_not_terminated]["avail_actions"][:, self.t],
                    t_env=self.t_env,
                    test_mode=True,
                    # bs=envs_not_terminated
                )
            cpu_actions = actions.to("cpu").numpy()

            # 3. 更新动作
            actions_chosen = {
                "actions": actions.unsqueeze(1)
            }
            self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # 4. 环境步进
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated and not terminated[idx]:
                    parent_conn.send(("step", cpu_actions[action_idx]))
                    action_idx += 1

            # 5. 收集环境反馈
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
                        # 保存最后一步的env_info
                        final_env_infos[idx] = data["info"]
                        if not pretrain and self.current_mcts_data[idx] is not None:
                            # 这样是不是可以保留batch的第一个维度？
                            final_input = self.mac._build_inputs(self.batch[idx], t=self.t, task=self.task, use_skill=True).reshape(1, self.n_agents, -1)
                            final_state = data["state"][np.newaxis, :]  # Add batch dimension
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
                    pre_transition_data["avail_skills"].append(np.ones((self.n_agents, self.args.skill_dim)))
            # 6. 更新batch
            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)
            self.t += 1
            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)
            # 7. 更新envs_not_terminated
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            if not envs_not_terminated:
                break

        if not test_mode:
            self.t_env += self.env_steps_this_run

        # 直接返回所有环境的replay buffer列表
        replay_buffers = self.replay_buffers_mcts if not pretrain else None

        # 记录统计信息，与episode_runner对齐
        result_info = {}
        if not pretrain:
            cur_stats = self.test_stats if test_mode else self.train_stats
            cur_returns = self.test_returns if test_mode else self.train_returns
            log_prefix = f"{self.task}/test_" if test_mode else f"{self.task}/"
            # 用final_env_infos替换原有的env_stats
            infos = [cur_stats] + [info for info in final_env_infos if info is not None]
            # 合并所有info的key
            all_keys = set()
            for d in infos:
                all_keys |= set(d)
            # 统计所有key的和
            merged_stats = {k: sum(d.get(k, 0) for d in infos) for k in all_keys}
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
                    self.logger.log_stat(f"{self.task}/epsilon", self.mac.action_selector.epsilon, self.t_env)
                self.log_train_stats_t = self.t_env

            # 统计返回
            avg_return = float(np.mean(episode_returns)) if episode_returns else 0.0
            avg_length = float(np.mean(episode_lengths)) if episode_lengths else 0.0
            # 取第一个非None的final_env_info
            stats_info = {}
            # 统计 battle_won 的平均值
            wins = [info.get("battle_won", 0) for info in final_env_infos if info is not None]
            avg_win = float(sum(wins)) / len(wins) if wins else 0.0

            # 统计所有 final_env_infos 中各项指标的平均值作为 stats_info
            stats_info = {}
            # 收集所有非 None 的 info 的键
            keys = set(k for info in final_env_infos if info for k in info)
            # 计算每个键的平均值
            for k in keys:
                vals = [info.get(k, 0) for info in final_env_infos if info is not None and k in info]
                stats_info[k] = float(sum(vals) / len(vals)) if vals else 0.0

            result_info = {
                "episode_return": avg_return,
                "win_rate": avg_win,
                "episode_length": avg_length,
                "stats": stats_info
            }
        else:
            result_info = {}

        return self.batch, replay_buffers, result_info

    def _batch_select_skill_with_mcts(self, env_indices):
        # env_indices: 需要新skill的环境下标
        batch_size = self.batch_size  # 使用所有环境
        # 构造batch输入（全部环境）
        state_inputs = []
        obs_inputs = []
        wm_hidden_states = []
        for idx in range(self.batch_size):
            state = self.batch["state"][idx:idx+1, self.t].reshape(1, -1).cpu().numpy()
            obs = self.mac.preprocess_obs(self.batch[idx], self.t, self.task, use_skill=True).cpu().numpy()
            state_inputs.append(state)
            obs_inputs.append(obs)
            wm_hidden_states.append(self.wm_hidden_states[idx])
        # 拼接为batch
        state_inputs = np.concatenate(state_inputs, axis=0)  # [B, state_dim]
        obs_inputs = np.concatenate(obs_inputs, axis=0)      # [B, n_agents, obs_dim]
        wm_hidden_states = th.stack(wm_hidden_states, axis=0)  # [B, ...]
        # 批量初始化root
        roots, experienced_thresholds, root_policy_hidden_states, root_critic_hidden_states = initialize_root(
            self.mcts_network,
            state_inputs,
            obs_inputs,
            self.k,
            self.n_agents,
            self.args.skill_dim,
            wm_hidden_states=wm_hidden_states,
            bs_id=list(range(self.batch_size)),
        )
        # rng_key, split_key = jax.random.split(self.rng_key)
        # 批量运行MCTS
        policy_output, timing_stats, advantages, new_wm_hidden_states, new_policy_hidden_states, new_critic_hidden_states, rng_key = mctx.gumbel_muzero_policy(
            params=(),
            rng_key=self.rng_key,
            root=roots,
            recurrent_fn=self.recurrent_fn,
            action_selection_fn=self.action_selection_fn,
            num_simulations=self.num_simulations,
            task=self.task,
            current_t_env=self.t_env,
            args=self.args,
            max_num_considered_actions=self.max_num_considered_actions,
            max_depth=None,
            qtransform=functools.partial(
                mctx.qtransform_completed_by_mix_value,
                use_mixed_value=self.use_mixed_value,
            ),
        )
        # 只挑选env_indices对应的结果
        self.rng_key = rng_key
        skill_indices = []  # [len(env_indices)]
        mcts_datas = []
        new_wm_hidden_states = th.tensor(np.array(new_wm_hidden_states)).to(self.args.device)
        new_policy_hidden_states = th.tensor(np.array(new_policy_hidden_states)).to(self.args.device)
        new_critic_hidden_states = th.tensor(np.array(new_critic_hidden_states)).to(self.args.device)
        for i, idx in enumerate(env_indices):
            # 这里的skill_indices是一个一维数组，长度为1
            skill_indices.append(policy_output[idx].chosen_skill)
            mcts_datas.append((
                policy_output[idx],
                experienced_thresholds[idx:idx+1],
                advantages[idx:idx+1],
                root_policy_hidden_states[idx:idx+1],
                root_critic_hidden_states[idx:idx+1]
            ))
    
        # selected_wm_hidden_states = new_wm_hidden_states[env_indices]
        # selected_policy_hidden_states = new_policy_hidden_states[env_indices]
        # selected_critic_hidden_states = new_critic_hidden_states[env_indices]
        # return skill_indices, mcts_datas, selected_wm_hidden_states, selected_policy_hidden_states, selected_critic_hidden_states
        skill_indices = np.array(skill_indices).reshape(len(env_indices), -1)
        return skill_indices, mcts_datas, new_wm_hidden_states, new_policy_hidden_states, new_critic_hidden_states


    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()


def env_worker(remote, env_fn):
    # 创建环境
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            # 在环境中执行一步
            reward, terminated, env_info = env.step(actions)
            # 返回观测、可用动作和状态以便选择下一个动作
            state = env.get_state()
            avail_actions = env.get_avail_actions()
            obs = env.get_obs()
            remote.send({
                # 下一时间步选择动作所需的数据
                "state": state,
                "avail_actions": avail_actions,
                "obs": obs,
                # 当前时间步的其余数据
                "reward": reward,
                "terminated": terminated,
                "info": env_info
            })
        elif cmd == "reset":
            env.reset()
            remote.send({
                "state": env.get_state(),
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
    """
    使用cloudpickle来序列化内容（否则多进程尝试使用pickle）
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        self.x = pickle.loads(ob)