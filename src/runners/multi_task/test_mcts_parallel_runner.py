import numpy as np
import torch as th
import copy
import sys
import os
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
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号
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
import jax.numpy as jnp
import torch.nn.functional as F
import copy
jax.config.update('jax_debug_nans', True)
# 配置日志级别，减少调试信息
logging.getLogger('jax').setLevel(logging.INFO)
logging.getLogger('absl').setLevel(logging.WARNING)

class TestMCTSParallelRunner:

    def __init__(self, args, logger, task):
        # ...现有代码保持不变...
        self.bandit_run_count = 0
        self.bandit_stats_history = {
            "mean_advantage": [],
            "mean_selected_action_value": [],
            "mean_prior_policy_action_value": [],
            "mean_action_weights_policy_value": [],
            "mean_root_value": [],  # 新增：根节点value值
            "mean_q_value_advantage": [],  # 新增：Q值相对于value的优势
            "prior_action_probs": [],  # 先验概率分布
            "mcts_action_weights": []  # 新增：MCTS搜索后的动作权重（实际概率）
        }
        # ...其余代码不变...
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
        self.task_decomposer = self.mac.task2decomposer[self.task]
        self.n_enemy = self.task_decomposer.n_enemies
        self.n_ally = self.n_agents - 1
        
        # 创建递归函数
        self.recurrent_fn = make_recurrent_fn_world_model(
            self.mac, 
            self.batch_size,  # 每个环境实例单独处理，所以batch_size=1
            self.mcts_network,
            self.temperature,
            self.n_agents,
            self.k,
            offline_value_start=self.args.offline_value_start,
            offline_value_end=self.args.offline_value_end,
            offline_value_anneal_time=self.args.offline_value_anneal_time
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
        # TODO:self.mctx_network.reset()是不是没有进行？
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
        # 并行相关状态变量初始化
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
        self.mac.init_hidden(batch_size=self.batch_size, task=self.task)
        # self.wm_hidden_states = self.mac.init_hidden_wm(batch_size=self.batch_size, task=self.task)
        self.mcts_network.init_hidden(batch_size=self.batch_size)
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        # 用于存储每个环境的最终env_info
        final_env_infos = [None for _ in range(self.batch_size)]
        predict_reward = []
        true_reward = []
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
                    # 全部赋值影响不大。因为各个不同的batch之间的hidden state都是相互不影响的
                    # self.wm_hidden_states = new_wm_hidden_states
                    self.mcts_network.policy_hidden = new_policy_hidden_states
                    self.mcts_network.critic_hidden = new_critic_hidden_states

            # 2. 批量动作选择
            if pretrain:
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=0, task=self.task, bs=envs_not_terminated, test_mode=False)
            else:   
                # 已经存在占位了，不会报错
                # skill_indices = [self.current_skill_indices[idx] for idx in envs_not_terminated]
                skill_indices = self.current_skill_indices
                # 这里一定要讲bs=envs_not_terminated传入，因为设计选取hidden state
                # 对于q mode（目前的mode），test_mode参数无影响，输出的是q值。如果是pi_logit mode，那么就没有随机性。

                # ------------------------这里是测试部分------------------------------------
                # TODO: 暂时使用select_actions获取skill_indices
                # actions, skill_indices = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, task=self.task, bs=envs_not_terminated, test_mode=True)
                # print(skill_indices)
                # skill_indices = [skill_indices.detach().cpu().numpy()]
                if getattr(self.args, "basic_action_as_skill", False):
                    actions = th.tensor(self.current_skill_indices, dtype=th.int64)[envs_not_terminated]

                else:
                    if not self.args.use_origin_model:
                        actions = self.mac.forward_action_skill(
                            self.batch,
                            t=self.t,
                            skill_index=skill_indices,
                            task=self.task,
                            # bs=envs_not_terminated,
                            test_mode=True,
                        )
                        avail_actions = self.batch["avail_actions"][:, self.t]
                        # 选择基本动作时，不需要随机性，直接将skill解码
                        actions = self.mac.action_selector.select_action(
                            actions[envs_not_terminated],
                            avail_actions[envs_not_terminated],
                            t_env=self.t_env,
                            test_mode=True,
                            # bs=envs_not_terminated
                        )
                # ------------------------------------------------------------
                # actions, skill_index = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, task=self.task, bs=envs_not_terminated, test_mode=True)
                # ------------------------------------------------------------
            
            # if getattr(self.args, "basic_action_as_skill", False):
            #     cpu_actions = actions.astype(np.int64)
            # else:
            cpu_actions = actions.to("cpu").numpy()

            # 3. 更新动作

            # 在更新 batch 之前，对 actions 和 skills 做 one‐hot 编码
            # bs 是当前未终止的环境 idx 列表
            action_idx = actions.squeeze(1)
            skill_idx = th.tensor([self.current_skill_indices[i] for i in envs_not_terminated]).squeeze()

            # one-hot 编码
            num_actions = self.batch["avail_actions"].shape[-1]
            # TODO: check here
            actions_onehot = F.one_hot(action_idx, num_classes=num_actions).float().unsqueeze(1)
            # 这个skills_onehot完全没用，因为不同t对应的skill完全不一样
            skills_onehot  = F.one_hot(skill_idx.long(), num_classes=self.args.skill_dim).float().unsqueeze(1)

            actions_chosen = {
                "actions":        actions.unsqueeze(1),
                "actions_onehot": actions_onehot,
                "skills_onehot":  skills_onehot
            }
            
            self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # 4. 环境步进
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated:
                    if not terminated[idx]:
                        parent_conn.send(("step", cpu_actions[action_idx]))
                    action_idx += 1
            # Update envs_not_terminated
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            all_terminated = all(terminated)
            if all_terminated:
                break
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
                            # TODO: 这里本来final_input应该是指data['obs']，但是data['obs']其实在最后的buffer里面并没有使用，所以这里就随便给一个进行替代了
                            final_input = self.mac._build_inputs(self.batch, t=self.t, task=self.task, use_skill=True).reshape(1, self.n_agents, -1)
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
                    if getattr(self.args, "basic_action_as_skill", False):
                        pre_transition_data["avail_skills"].append(data["avail_actions"])
                    else:
                        pre_transition_data["avail_skills"].append(np.ones((self.n_agents, self.args.skill_dim)))
            # 6. 更新batch
            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)
            self.t += 1
            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)

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
        predict_reward = np.array(predict_reward)
        true_reward = np.array(true_reward)
        mse = np.mean((predict_reward - true_reward) ** 2)
        print(f"MSE: {mse}")
        # a = np.abs(predict_reward - true_reward)/(true_reward+predict_reward)
        # print(np.mean(a))
        return self.batch, replay_buffers, result_info

    def save_initial_state(self):
        """保存初始状态，用于后续的固定状态测试"""
        self.reset()
        
        # 保存初始batch、环境状态和hidden states
        self.bandit_batch = copy.deepcopy(self.batch)
        self.bandit_t = 0
        
        # 初始化隐藏状态
        self.mac.init_hidden(batch_size=self.batch_size, task=self.task)
        self.mcts_network.init_hidden(batch_size=self.batch_size)
        
        # 保存隐藏状态
        
        self.bandit_policy_hidden = self.mcts_network.policy_hidden.clone()
        self.bandit_critic_hidden = self.mcts_network.critic_hidden.clone()
        
        # 如果wm_hidden_states存在，也保存它
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
        """在固定状态上测试MCTS搜索的有效性"""
        if not hasattr(self, 'bandit_batch'):
            print("请先调用save_initial_state()保存初始状态")
            return None, None, {}
        
        # 创建一个新的replay buffer来保存结果
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
        root_values = []  # 新增：收集根节点的value
        q_value_advantages = []  # 新增：Q值相对于value的优势
        prior_probs_list = []  # 新增：收集prior_logits转换为的概率分布
        mcts_action_weights_list = []  # 新增：收集MCTS搜索后的动作权重
        
        # 重置统计数据
        bandit_stats = {
            "mean_advantage": 0.0,
            "mean_selected_action_value": 0.0,
            "mean_prior_policy_action_value": 0.0,
            "action_weight_policy_value": 0.0,
        }
        
        for trial in range(num_trials):
            # 恢复保存的状态
            batch = copy.deepcopy(self.bandit_batch)
            
            # 恢复隐藏状态
            
            self.mcts_network.policy_hidden = self.bandit_policy_hidden.clone()
            self.mcts_network.critic_hidden = self.bandit_critic_hidden.clone()
            
            if self.bandit_wm_hidden is not None:
                if hasattr(self.mac, 'hidden_states_reward') and hasattr(self.mac, 'hidden_states_value'):
                    self.mac.hidden_states_reward = self.bandit_wm_hidden[:, 0, :].clone()
                    self.mac.hidden_states_value = self.bandit_wm_hidden[:, 1, :].clone()
                self.wm_hidden_states = self.bandit_wm_hidden.clone()
                
            # 构建状态和观察输入
            state_inputs = []
            obs_inputs = []
            
            for idx in range(self.batch_size):
                state = batch["state"][idx:idx+1, self.bandit_t].reshape(1, -1).cpu().numpy()
                obs = self.mac.preprocess_obs(batch[idx], self.bandit_t, self.task, use_skill=True).cpu().numpy()
                state_inputs.append(state)
                obs_inputs.append(obs)
                
            # 拼接为batch
            state_inputs = np.concatenate(state_inputs, axis=0)  # [B, state_dim]
            obs_inputs = np.concatenate(obs_inputs, axis=0)      # [B, n_agents, obs_dim]
            
            # 使用mac进行价值预测
            value_pre = self.mac.forward_value(
                batch,
                t=self.bandit_t,
                task=self.task)
            value_pre.unsqueeze_(1)
            state_input = batch["state"][:, self.bandit_t]
            state_input.unsqueeze_(1)
            outer_value = self.mac.mixer(
                value_pre, state_input, self.task_decomposer
            )
            
            # 初始化root
            roots, experienced_thresholds, root_policy_hidden_states, root_critic_hidden_states = initialize_root(
                self.mcts_network,
                state_inputs,
                obs_inputs,
                self.k,
                self.n_agents,
                self.args.skill_dim,
                wm_hidden_states=self.wm_hidden_states,
                bs_id=list(range(self.batch_size)),
                # TODO: just for test. outer value作为一个固定不变的评估，和使用一个变化的评估，有什么不一样
                outer_value=outer_value,
                avail_skills=batch["avail_actions"][:, self.bandit_t] if getattr(self.args, "basic_action_as_skill", False) else None,
            )
            
            # 进行MCTS搜索
            policy_output, timing_stats, advantages_data, new_wm_hidden_states, new_policy_hidden_states, new_critic_hidden_states, rng_key = mctx.gumbel_muzero_policy(
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
                qtransform=functools.partial(
                    mctx.qtransform_completed_by_mix_value,
                    use_mixed_value=self.use_mixed_value,
                ),
            )
            
            self.rng_key = rng_key
            batch_range = jnp.arange(self.batch_size)
            root_idx = 0
            # 收集数据以计算优势
            # for b_idx in range(self.batch_size):
            # 获取搜索树的值和逻辑
            # TODO: 这里应该是怎么处理，后面再看吧
            # 在搜索树数据收集部分
            qvalues = policy_output.search_tree.qvalues(batch_range)
            prior_logits = policy_output.search_tree.children_prior_logits[batch_range, root_idx]
            
            # 计算prior_logits的softmax，得到动作概率分布
            prior_probs = jax.nn.softmax(prior_logits, axis=-1)
            # 明确转换为numpy数组
            prior_probs_list.append(np.array(prior_probs))
            
            # 获取选定的动作及其值
            selected_action = policy_output.action
            selected_action_value = qvalues[batch_range, root_idx, selected_action]
            
            # 获取先验策略的动作
            gumbel = policy_output.search_tree.extra_data.root_gumbel
            prior_policy_action = jnp.argmax(gumbel + prior_logits, axis=-1)
            
            # 获取根节点value
            root_value = roots.value
            prior_policy_action_value = qvalues[batch_range, root_idx, prior_policy_action]
            
            # 计算Q值相对于value的优势
            q_value_advantage = selected_action_value - root_value
            
            print("prior action: {}, selected action: {}".format(prior_policy_action, selected_action))
            print("root value: {:.4f}, selected Q value: {:.4f}, Q-V advantage: {:.4f}".format(
                jnp.mean(root_value).item(), 
                jnp.mean(selected_action_value).item(),
                jnp.mean(q_value_advantage).item()
            ))
            # 计算带权重的策略值
            action_weights = policy_output.action_weights
            # 明确转换为numpy数组
            mcts_action_weights_list.append(np.array(action_weights))
            action_weights_policy_value = jnp.sum(action_weights * qvalues[batch_range, root_idx], axis=-1)
            
            # 存储数据
            prior_actions.append(prior_policy_action)
            mcts_actions.append(selected_action)
            advantages.append(selected_action_value - prior_policy_action_value)
            root_values.append(root_value)  # 新增
            q_value_advantages.append(q_value_advantage)  # 新增
            qvalues_data.append({
                "selected_action": selected_action,
                "selected_action_value": selected_action_value,
                "prior_action": prior_policy_action,
                "prior_action_value": prior_policy_action_value,
                "action_weights_policy_value": action_weights_policy_value,
                "root_value": root_value,  # 新增
                "q_value_advantage": q_value_advantage  # 新增
            })
            for b_idx in range(self.batch_size):
                # 将结果保存到replay buffer
                bandit_replay_buffer.push(
                    policy_output[b_idx],
                    experienced_thresholds[b_idx:b_idx+1],
                    advantages_data[b_idx:b_idx+1],
                    root_policy_hidden_states[b_idx:b_idx+1],
                    root_critic_hidden_states[b_idx:b_idx+1],
                    # real_r=0.0,  # 没有真实奖励
                    # real_next_obs=None,  # 没有真实下一个观察
                    # real_next_state=None,  # 没有真实下一个状态
                    # real_done=np.zeros(1, dtype=bool)  # 没有终止
                )
        
        # 计算统计数据时添加新指标
        bandit_stats = {
            "mean_advantage": np.mean(advantages),
            "mean_selected_action_value": np.mean([q["selected_action_value"] for q in qvalues_data]),
            "mean_prior_policy_action_value": np.mean([q["prior_action_value"] for q in qvalues_data]),
            "mean_action_weights_policy_value": np.mean([q["action_weights_policy_value"] for q in qvalues_data]),
            "mean_root_value": np.mean([q["root_value"] for q in qvalues_data]),
            "mean_q_value_advantage": np.mean([q["q_value_advantage"] for q in qvalues_data]),
            # 确保转换为标准NumPy数组
            "prior_action_probs": np.mean(np.array(prior_probs_list), axis=0).astype(np.float64),
            "mcts_action_weights": np.mean(np.array(mcts_action_weights_list), axis=0).astype(np.float64)
        }
        
        # 打印结果时添加新指标
        print(f"MCTS测试结果 ({num_trials} 次尝试):")
        print(f"平均优势: {bandit_stats['mean_advantage']:.4f}")
        print(f"MCTS选择动作的平均值: {bandit_stats['mean_selected_action_value']:.4f}")
        print(f"先验策略动作的平均值: {bandit_stats['mean_prior_policy_action_value']:.4f}")
        print(f"带权重的策略值平均: {bandit_stats['mean_action_weights_policy_value']:.4f}")
        print(f"根节点value平均值: {bandit_stats['mean_root_value']:.4f}")  # 新增
        print(f"Q值相对于value的优势: {bandit_stats['mean_q_value_advantage']:.4f}")  # 新增

        # 新增：更新历史数据并绘制图表
        self.bandit_run_count += 1
        for key in self.bandit_stats_history:
            if key in bandit_stats:
                self.bandit_stats_history[key].append(bandit_stats[key])
        
        # 每10次运行绘制一次图表
        print(f"当前运行次数: {self.bandit_run_count}")
        if self.bandit_run_count % 10 == 0:
            self.plot_bandit_stats()
        
        return batch, bandit_replay_buffer, bandit_stats
    def _batch_select_skill_with_mcts(self, env_indices):
        # env_indices: 需要新skill的环境下标
        batch_size = self.batch_size  # 使用所有环境
        # 构造batch输入（全部环境）
        state_inputs = []
        obs_inputs = []
        # wm_hidden_states = []
        for idx in range(self.batch_size):
            state = self.batch["state"][idx:idx+1, self.t].reshape(1, -1).cpu().numpy()
            obs = self.mac.preprocess_obs(self.batch[idx], self.t, self.task, use_skill=True).cpu().numpy()
            state_inputs.append(state)
            obs_inputs.append(obs)
            # wm_hidden_states.append(self.wm_hidden_states[idx])
        if self.wm_hidden_states is None:
            hidden_states_reward = self.mac.hidden_states_reward.clone()  # [B, ...]
            hidden_states_value = self.mac.hidden_states_value.clone()  # [B, ...]
            hidden_states_reward = hidden_states_reward.unsqueeze(1)
            hidden_states_value = hidden_states_value.unsqueeze(1)
            wm_hidden_states = th.cat([hidden_states_reward, hidden_states_value], dim=1)
            self.wm_hidden_states = wm_hidden_states.clone()
        else:
            self.wm_hidden_states[:,1] = self.mac.hidden_states_value.reshape(self.batch_size, self.n_agents, -1)
            wm_hidden_states = self.wm_hidden_states.clone()
        # 拼接为batch
        state_inputs = np.concatenate(state_inputs, axis=0)  # [B, state_dim]
        obs_inputs = np.concatenate(obs_inputs, axis=0)      # [B, n_agents, obs_dim]
        # wm_hidden_states = th.stack(wm_hidden_states, axis=0)  # [B, ...]
        # TODO:just for test------------------------------------------------------
        # 这里可以使用mac进行预测，因为这个是真实的，无论如何也会发生
        value_pre = self.mac.forward_value(
            self.batch,
            t=self.t,
            task=self.task,)
        value_pre.unsqueeze_(1)
        state_input = self.batch["state"][:, self.t]
        state_input.unsqueeze_(1)
        outer_value = self.mac.mixer(
                value_pre, state_input, self.task_decomposer
            )
        # ------------------------------------------------------------------------
        # 将wm中的hidden state置换为真实的value预测后的hidden state
        # 不需要skill作出决定就能获取的，就在root（e.g., value hidden state, policy hidden state)
        # 需要skill才能获取的，不再root(e.g., reward hidden state)
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
            outer_value=outer_value,
            avail_skills=self.batch["avail_actions"][:, self.t] if getattr(self.args, "basic_action_as_skill", False) else None,
        )
        # rng_key, split_key = jax.random.split(self.rng_key)
        # 批量运行MCTS. 这里的new_wm_hidden_states, new_policy_hidden_states, new_critic_hidden_states其实就是initialize root里的hidden state
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
            # max_depth=2,
            qtransform=functools.partial(
                mctx.qtransform_completed_by_mix_value,
                use_mixed_value=self.use_mixed_value,
            ),
        )
        # 只挑选env_indices对应的结果
        self.rng_key = rng_key
        skill_indices = []  # [len(env_indices)]
        mcts_datas = []

        # ----------Post update: 有了动作之后，更新hidden_state_reward,因为reward predict需要和动作相关才能使用
        new_wm_hidden_states = th.tensor(np.array(new_wm_hidden_states)).to(self.args.device)
        # self.mac.hidden_states_reward = new_wm_hidden_states[:, 0, :]
        self.wm_hidden_states = new_wm_hidden_states
        # ----------
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

    def plot_bandit_stats(self, save_dir="./results/mcts_bandit"):
        """绘制bandit_stats历史数据并保存"""
        # import matplotlib.pyplot as plt
        # import os
        
        # # 配置中文字体支持
        # import matplotlib
        # matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'WenQuanYi Micro Hei', 'Arial Unicode MS']  # 用来正常显示中文
        # matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        
        # 确保目录存在
        os.makedirs(save_dir, exist_ok=True)
        
        if not self.bandit_stats_history["mean_advantage"]:
            print("没有可供绘制的历史数据")
            return
        
        # 创建图表
        plt.figure(figsize=(12, 8))
        fig, ax1 = plt.subplots(figsize=(12, 8))
        
        # 绘制每个指标
        x = list(range(1, len(self.bandit_stats_history["mean_advantage"])*10+1, 10))
        ax1.plot(x, self.bandit_stats_history["mean_advantage"], 'o-', label='mean_advantage', color='red')
        ax1.set_xlabel('test times', fontsize=12)
        ax1.set_ylabel('advantage', color='red', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='red')
        
        # 创建第二个y轴
        ax2 = ax1.twinx()
        ax2.plot(x, self.bandit_stats_history["mean_selected_action_value"], 's-', 
                label='selected_action_value', color='blue')
        ax2.plot(x, self.bandit_stats_history["mean_prior_policy_action_value"], '^-', 
                label='prior_policy_action_value', color='green')
        ax2.plot(x, self.bandit_stats_history["mean_action_weights_policy_value"], 'd-', 
                label='action_weights_policy_value', color='purple')
        ax2.set_ylabel('Q value', fontsize=12)
        
        # 合并两个坐标轴的图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
        
        plt.title(f'MCTS Bandit test result ({self.bandit_run_count} times)', fontsize=14)
        plt.grid(True)
        
        # 保存图表
        save_path = os.path.join(save_dir, f"mcts_bandit_stats_{self.bandit_run_count}.png")
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close(fig)  # 明确关闭图形，避免内存泄漏
        
        # 绘制prior_action_probs的柱状图
        if "prior_action_probs" in self.bandit_stats_history and len(self.bandit_stats_history["prior_action_probs"]) > 0:
            plt.figure(figsize=(12, 8))
            # 确保是标准NumPy数组并且是一维的
            latest_probs = np.array(self.bandit_stats_history["prior_action_probs"][-1]).flatten().astype(np.float64)
            
            # 绘制柱状图
            plt.bar(range(len(latest_probs)), latest_probs)
            plt.xlabel('动作索引', fontsize=12)
            plt.ylabel('先验概率', fontsize=12)
            plt.title(f'动作先验概率分布 (第{self.bandit_run_count}次测试)', fontsize=14)
            plt.grid(True, alpha=0.3)
            
            # 添加数值标签
            for i, prob in enumerate(latest_probs):
                if prob > 0.01:  # 只标注概率大于1%的动作
                    plt.text(i, prob, f'{prob:.3f}', ha='center', va='bottom', fontsize=9)
                    
            prior_probs_path = os.path.join(save_dir, f"prior_probs_{self.bandit_run_count}.png")
            plt.savefig(prior_probs_path, dpi=120, bbox_inches='tight')
            plt.close()
            print(f"先验概率分布图已保存至 {prior_probs_path}")
        
        # 绘制MCTS搜索后的动作权重分布（实际被选择概率）
        if "mcts_action_weights" in self.bandit_stats_history and len(self.bandit_stats_history["mcts_action_weights"]) > 0:
            plt.figure(figsize=(12, 8))
            # 确保是标准NumPy数组并且是一维的
            latest_weights = np.array(self.bandit_stats_history["mcts_action_weights"][-1]).flatten().astype(np.float64)
            
            # 绘制柱状图
            plt.bar(range(len(latest_weights)), latest_weights)
            plt.xlabel('动作索引', fontsize=12)
            plt.ylabel('MCTS搜索后概率', fontsize=12)
            plt.title(f'MCTS动作选择概率分布 (第{self.bandit_run_count}次测试)', fontsize=14)
            plt.grid(True, alpha=0.3)
            
            # 添加数值标签
            for i, weight in enumerate(latest_weights):
                if weight > 0.01:  # 只标注概率大于1%的动作
                    plt.text(i, weight, f'{weight:.3f}', ha='center', va='bottom', fontsize=9)
                    
            mcts_weights_path = os.path.join(save_dir, f"mcts_action_weights_{self.bandit_run_count}.png")
            plt.savefig(mcts_weights_path, dpi=120, bbox_inches='tight')
            plt.close()
            print(f"MCTS动作选择概率分布图已保存至 {mcts_weights_path}")
            
        # 添加比较图：同时展示先验概率和MCTS搜索后概率
        if ("prior_action_probs" in self.bandit_stats_history and 
            "mcts_action_weights" in self.bandit_stats_history and 
            len(self.bandit_stats_history["prior_action_probs"]) > 0 and 
            len(self.bandit_stats_history["mcts_action_weights"]) > 0):
            
            plt.figure(figsize=(14, 10))
            # 确保是标准NumPy数组并且是一维的
            prior_probs = np.array(self.bandit_stats_history["prior_action_probs"][-1]).flatten().astype(np.float64)
            mcts_weights = np.array(self.bandit_stats_history["mcts_action_weights"][-1]).flatten().astype(np.float64)
            
            x = np.arange(len(prior_probs))
            width = 0.35
            
            fig, ax = plt.subplots(figsize=(14, 10))
            rects1 = ax.bar(x - width/2, prior_probs, width, label='先验概率')
            rects2 = ax.bar(x + width/2, mcts_weights, width, label='MCTS搜索后概率')
            
            ax.set_xlabel('动作索引', fontsize=12)
            ax.set_ylabel('概率', fontsize=12)
            ax.set_title(f'动作概率对比 (第{self.bandit_run_count}次测试)', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 只对主要动作添加标签
            threshold = 0.05
            for i, (prior, mcts) in enumerate(zip(prior_probs, mcts_weights)):
                if prior > threshold or mcts > threshold:
                    if prior > threshold:
                        ax.text(i - width/2, prior, f'{prior:.2f}', ha='center', va='bottom', fontsize=8)
                    if mcts > threshold:
                        ax.text(i + width/2, mcts, f'{mcts:.2f}', ha='center', va='bottom', fontsize=8)
            
            compare_path = os.path.join(save_dir, f"action_probs_compare_{self.bandit_run_count}.png")
            plt.savefig(compare_path, dpi=120, bbox_inches='tight')
            plt.close()
            print(f"动作概率对比图已保存至 {compare_path}")
        
        print(f"图表已保存至 {save_path}")
        
        # 检查哪些字段在所有运行中都有数据并且长度一致
        basic_keys = ["mean_advantage", "mean_selected_action_value", 
                     "mean_prior_policy_action_value", "mean_action_weights_policy_value", 
                     "mean_root_value", "mean_q_value_advantage"]
        
        # 筛选有数据的键
        available_keys = []
        for key in basic_keys:
            if key in self.bandit_stats_history and len(self.bandit_stats_history[key]) > 0:
                available_keys.append(key)
        
        # 如果没有可用的数据，则退出
        if not available_keys:
            print("没有可用的历史数据进行CSV保存")
            return
        
        # 使用最小长度来限制数据
        min_length = min(len(self.bandit_stats_history[key]) for key in available_keys)
        x = list(range(1, min_length*10+1, 10))
        
        # 创建DataFrame，只包含有效数据
        data = {"run_count": x}
        for key in available_keys:
            data[key] = self.bandit_stats_history[key][:min_length]
        
        # 生成一个包含所有数据的CSV文件
        import pandas as pd
        df = pd.DataFrame(data)
        csv_path = os.path.join(save_dir, "mcts_bandit_stats.csv")
        df.to_csv(csv_path, index=False)
        
        # 保存最新的先验概率分布和MCTS动作权重到单独的CSV文件
        if "prior_action_probs" in self.bandit_stats_history and len(self.bandit_stats_history["prior_action_probs"]) > 0:
            latest_probs = np.array(self.bandit_stats_history["prior_action_probs"][-1]).flatten().astype(np.float64)
            probs_df = pd.DataFrame(latest_probs, columns=["prior_prob"])
            probs_csv_path = os.path.join(save_dir, f"prior_action_probs_{self.bandit_run_count}.csv")
            probs_df.to_csv(probs_csv_path, index=False)
            print(f"最新的先验概率分布已保存至 {probs_csv_path}")
        
        if "mcts_action_weights" in self.bandit_stats_history and len(self.bandit_stats_history["mcts_action_weights"]) > 0:
            latest_weights = np.array(self.bandit_stats_history["mcts_action_weights"][-1]).flatten().astype(np.float64)
            weights_df = pd.DataFrame(latest_weights, columns=["mcts_action_weight"])
            weights_csv_path = os.path.join(save_dir, f"mcts_action_weights_{self.bandit_run_count}.csv")
            weights_df.to_csv(weights_csv_path, index=False)
            print(f"最新的MCTS动作权重已保存至 {weights_csv_path}")
        
        print(f"数据已保存至 {csv_path}")

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
    """
    使用cloudpickle来序列化内容（否则多进程尝试使用pickle）
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        self.x = pickle.loads(ob)