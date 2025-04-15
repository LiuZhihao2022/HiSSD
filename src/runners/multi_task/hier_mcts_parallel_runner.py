import numpy as np
import torch as th
import copy
import sys
import logging
import pickle
import cloudpickle
import functools
# import mctx

from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe, Process

# FROM policy_improvement_demo.py
sys.path.append('/home/liuzhihao/HiSSD')
from typing import Tuple, Optional
from absl import app
from absl import flags
from mctx._src.network import PolicyRNN, ReplayBuffer, compute_prior_from_qvalues
from mctx._src.optimizer_wrapper import ValueOptimizerWrapper
from mctx._src.simple_env import SimpleEnv
from mctx._src.recurrent_fn import make_recurrent_fn_gym, make_recurrent_fn_world_model, make_multiagent_recurrent_fn_gym
from mctx._src.utils import stochastic_top_k_sampling
from examples.policy_improvement_demo import initialize_root, DemoOutput

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
        self.num_simulations = args.num_simulations if hasattr(args, "num_simulations") else 32
        self.max_num_considered_actions = args.max_num_considered_actions if hasattr(args, "max_num_considered_actions") else 16
        self.use_mixed_value = args.use_mixed_value if hasattr(args, "use_mixed_value") else False
        self.k = args.k if hasattr(args, "k") else 10
        self.temperature = args.temperature if hasattr(args, "temperature") else 1.0

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
        self.wm_hidden_states = [self.mac.init_hidden_wm(batch_size=1, task=self.task) 
                               for _ in range(self.batch_size)]
        
        # 创建递归函数
        self.recurrent_fn = make_recurrent_fn_world_model(
            self.mac, 
            1,  # 每个环境实例单独处理，所以batch_size=1
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
        
        # 重置MCTS相关数据
        self.wm_hidden_states = [self.mac.init_hidden_wm(batch_size=1, task=self.task) 
                               for _ in range(self.batch_size)]
        
        self.replay_buffers_mcts = [ReplayBuffer(
            self.episode_limit//self.c_step + 1,
            1,
            self.c_step,
            use_real_data=True
        ) for _ in range(self.batch_size)]
        
        self.current_skill_indices = [None for _ in range(self.batch_size)]
        self.current_mcts_data = [None for _ in range(self.batch_size)]
        self.last_skill_selection_t = [0 for _ in range(self.batch_size)]
        self.accumulated_rewards = [0 for _ in range(self.batch_size)]
        self.last_observations = [None for _ in range(self.batch_size)]
        self.last_states = [None for _ in range(self.batch_size)]

    def run(self, test_mode=False, pretrain_phase=False):
        self.reset()

        all_terminated = False
        episode_returns = [0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        self.mac.init_hidden(batch_size=self.batch_size, task=self.task)
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        final_env_infos = []  # 按终止顺序存储额外信息

        while True:
            # 为未终止的环境选择技能（如果需要）
            if not pretrain_phase:
                for idx in envs_not_terminated:
                    current_t = episode_lengths[idx]
                    # 每隔c_step步或者初始状态时选择技能
                    if current_t % self.c_step == 0:
                        # 存储上一个周期的MCTS数据（如果有）
                        if self.current_mcts_data[idx] is not None and current_t > 0:
                            current_input = self.mac._build_inputs(self.batch, t=self.t, task=self.task, bs=[idx]).reshape(1, self.n_agents, -1)
                            current_state = self.batch["state"][idx:idx+1, self.t]
                            
                            policy_output, experienced_thresholds, advantages, root_policy_hidden_state, root_critic_hidden_state = self.current_mcts_data[idx]
                            self.replay_buffers_mcts[idx].push(
                                policy_output, 
                                experienced_thresholds, 
                                advantages, 
                                root_policy_hidden_state, 
                                root_critic_hidden_state,
                                real_r=self.accumulated_rewards[idx],
                                real_next_obs=current_input,
                                real_next_state=current_state,
                                real_done=np.zeros(1, dtype=bool)  # 中间步骤不是终止状态
                            )
                            self.accumulated_rewards[idx] = 0
                        
                        # 使用MCTS选择技能
                        mcts_results = self._select_skill_with_mcts(idx)
                        self.current_skill_indices[idx] = mcts_results[0]
                        self.current_mcts_data[idx] = mcts_results[1:]
                        self.last_skill_selection_t[idx] = current_t
                        current_input = self.mac._build_inputs(self.batch, t=self.t, task=self.task, bs=[idx]).reshape(1, self.n_agents, -1)
                        self.last_observations[idx] = current_input
                        self.last_states[idx] = self.batch["state"][idx:idx+1, self.t]
            
            # 根据当前选择的技能或直接选择动作
            if pretrain_phase:
                # 预训练阶段，随机选择动作
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=0, task=self.task, bs=envs_not_terminated, test_mode=False)
            else:
                # 使用当前选择的技能来选择动作
                skill_indices = [self.current_skill_indices[idx] for idx in envs_not_terminated]
                actions = self.mac.forward_action_skill_batch(
                    self.batch,
                    t=self.t,
                    skill_indices=skill_indices,
                    task=self.task,
                    bs=envs_not_terminated,
                    test_mode=test_mode,
                )
                
                # 根据可用动作选择实际动作
                actions = self.mac.action_selector.select_action(
                    actions,
                    self.batch["avail_actions"][:, self.t],
                    t_env=self.t_env,
                    test_mode=test_mode,
                    bs=envs_not_terminated
                )
            
            cpu_actions = actions.to("cpu").numpy()

            # 更新选择的动作
            actions_chosen = {
                "actions": actions.unsqueeze(1)
            }
            self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # 向每个环境发送动作
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated:  # 为该环境生成了动作
                    if not terminated[idx]:  # 只有未终止的环境才发送动作
                        parent_conn.send(("step", cpu_actions[action_idx]))
                    action_idx += 1  # 递增动作索引

            # 更新未终止环境列表
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            all_terminated = all(terminated)
            if all_terminated:
                break

            # 当前时间步要插入的数据
            post_transition_data = {
                "reward": [],
                "terminated": []
            }
            # 下一时间步要插入的数据
            pre_transition_data = {
                "state": [],
                "avail_actions": [],
                "avail_skills": [],
                "obs": []
            }

            # 接收每个未终止环境的数据
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()
                    # 当前时间步的剩余数据
                    post_transition_data["reward"].append((data["reward"],))

                    episode_returns[idx] += data["reward"]
                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1
                    
                    # 累积当前技能周期的奖励
                    if not pretrain_phase:
                        self.accumulated_rewards[idx] = self.args.gamma * self.accumulated_rewards[idx] + data["reward"]

                    env_terminated = False
                    if data["terminated"]:
                        final_env_infos.append(data["info"])
                        
                        # 环境终止时，存储最后一个MCTS周期数据（如果有）
                        if not pretrain_phase and self.current_mcts_data[idx] is not None:
                            final_input = self.mac._build_inputs(self.batch, t=self.t, task=self.task, bs=[idx]).reshape(1, self.n_agents, -1)
                            final_state = data["state"]
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
                                real_done=np.ones(1, dtype=bool)  # 结束状态
                            )
                            
                    if data["terminated"] and not data["info"].get("episode_limit", False):
                        env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))

                    # 下一时间步选择动作需要的数据
                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["obs"].append(data["obs"])
                    pre_transition_data["avail_skills"].append(np.ones((self.n_agents, self.args.skill_dim)))

            # 将数据添加到batch中
            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # 进入下一个时间步
            self.t += 1

            # 添加预过渡数据
            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)

        if not test_mode:
            self.t_env += self.env_steps_this_run

        # 获取每个环境的统计信息
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_stats", None))

        env_stats = []
        for parent_conn in self.parent_conns:
            env_stat = parent_conn.recv()
            env_stats.append(env_stat)

        # 合并所有环境的MCTS回放缓冲区
        combined_replay_buffer = None
        if not pretrain_phase:
            # 创建一个足够大的缓冲区来存储所有数据
            max_steps = max([buffer.step for buffer in self.replay_buffers_mcts if buffer.step > 0], default=0)
            if max_steps > 0:
                combined_replay_buffer = ReplayBuffer(
                    max_steps,
                    self.batch_size,
                    self.c_step,
                    use_real_data=True
                )
                # 合并各个环境的回放缓冲区数据
                for idx, buffer in enumerate(self.replay_buffers_mcts):
                    if buffer.step > 0:
                        for i in range(buffer.step):
                            data = buffer.get_data(i)
                            combined_replay_buffer.push(
                                data['policy_output'],
                                data['experienced_thresholds'],
                                data['advantages'],
                                data['root_policy_hidden_state'],
                                data['root_critic_hidden_state'],
                                real_r=data['real_r'],
                                real_next_obs=data['real_next_obs'],
                                real_next_state=data['real_next_state'],
                                real_done=data['real_done']
                            )

        # 记录统计信息
        if not pretrain_phase:        
            cur_stats = self.test_stats if test_mode else self.train_stats
            cur_returns = self.test_returns if test_mode else self.train_returns
            log_prefix = f"{self.task}/test_" if test_mode else f"{self.task}/"
            infos = [cur_stats] + final_env_infos
            cur_stats.update({k: sum(d.get(k, 0) for d in infos) for k in set.union(*[set(d) for d in infos])})
            cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
            cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

            cur_returns.extend(episode_returns)

            n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
            if test_mode and (len(self.test_returns) == n_test_runs):
                self._log(cur_returns, cur_stats, log_prefix)
            elif not test_mode and self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
                self._log(cur_returns, cur_stats, log_prefix)
                if hasattr(self.mac.action_selector, "epsilon"):
                    self.logger.log_stat(f"{self.task}/epsilon", self.mac.action_selector.epsilon, self.t_env)
                self.log_train_stats_t = self.t_env

        return self.batch, combined_replay_buffer

    def _select_skill_with_mcts(self, env_idx):
        """
        为特定环境实例使用MCTS选择技能
        
        Args:
            env_idx: 环境实例的索引
            
        Returns:
            tuple: 包含技能选择结果的元组
        """
        # 获取当前状态和观测
        state_inputs = self.batch["state"][env_idx:env_idx+1, self.t]
        # 获取上一步的动作
        batch_last_action = None
        if self.t > 0:
            batch_last_action = self.batch["actions_onehot"][env_idx:env_idx+1, self.t - 1]
        
        # 处理状态输入
        state_inputs = state_inputs.reshape(1, -1).cpu().numpy()
        # 处理观测输入
        obs_inputs = self.mac.preprocess_obs(self.batch, self.t, self.task, bs=[env_idx]).cpu().numpy()
        
        # 初始化根节点
        root, experienced_thresholds, root_policy_hidden_state, root_critic_hidden_state = initialize_root(
            self.mcts_network, 
            state_inputs, 
            obs_inputs, 
            self.k,
            self.n_agents,
            self.args.skill_dim,
            wm_hidden_states=self.wm_hidden_states[env_idx]
        )
        
        # 运行MCTS搜索
        policy_output, timing_stats, advantages = mctx.gumbel_muzero_policy(
            params=(),
            rng_key=np.random.RandomState(),
            root=root,
            recurrent_fn=self.recurrent_fn,
            num_simulations=self.num_simulations,
            task=self.task,
            max_num_considered_actions=self.max_num_considered_actions,
            max_depth=None,
            qtransform=functools.partial(
                mctx.qtransform_completed_by_mix_value,
                use_mixed_value=self.use_mixed_value,
            ),
        )
        
        # 更新world model的隐藏状态
        self.wm_hidden_states[env_idx] = root_policy_hidden_state
        
        # 返回选择的技能和相关数据
        return policy_output.chosen_skill, policy_output, experienced_thresholds, advantages, root_policy_hidden_state, root_critic_hidden_state

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