from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np

# FROM policy_improvement_demo.py
import sys
sys.path.append('/home/liuzhihao/HiSSD')
import functools
from typing import Tuple, Optional
import chex
from absl import app
from absl import flags
import mctx
from mctx._src.network import PolicyRNN, ReplayBuffer, compute_prior_from_qvalues
from mctx._src.optimizer_wrapper import ValueOptimizerWrapper
from mctx._src.simple_env import SimpleEnv
from mctx._src.recurrent_fn import make_recurrent_fn_gym, make_recurrent_fn_world_model, make_multiagent_recurrent_fn_gym
# from mctx._src.utils import convert_tree_to_graph, stochastic_top_k_sampling
from mctx._src.utils import stochastic_top_k_sampling
from examples.policy_improvement_demo import initialize_root, DemoOutput

class HierMCTSEpisodeRunner:

    def __init__(self, args, logger, task):
        self.args = args
        self.logger = logger
        self.task = task
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        if args.env == "sc2":
            self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        elif args.env == "gymma":
            self.env = env_REGISTRY[self.args.env](
                **self.args.env_args,
                common_reward=self.args.common_reward,
                reward_scalarisation=self.args.reward_scalarisation,
            )
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0  # 跟踪总环境交互步数

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000
        
        # MCTS相关参数
        self.c_step = args.c_step
        self.num_simulations = args.num_simulations if hasattr(args, "num_simulations") else 32
        self.max_num_considered_actions = args.max_num_considered_actions if hasattr(args, "max_num_considered_actions") else 16
        self.use_mixed_value = args.use_mixed_value if hasattr(args, "use_mixed_value") else False
        self.k = args.k if hasattr(args, "k") else 10
        self.temperature = args.temperature if hasattr(args, "temperature") else 1.0
        self.current_skill_index = None
        self.wm_hidden_states = None

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
        # wm_hidden_states的更新要根据选择的skill_index来更新
        self.wm_hidden_states = self.mac.init_hidden_wm(batch_size=self.batch_size, task=self.task)
        self.n_agents = self.mac.task2n_agents[self.task]
        task_decomposer = self.mac.task2decomposer[self.task]
        self.n_enemy = task_decomposer.n_enemies
        self.n_ally = self.n_agents - 1
        # TODO: 参数需要后续来确定一下
        # TODO： obs的数目和agent的数目对不上，这里怎么处理？
        # TODO: 有没有把obs处理成每一个agent单独的obs的代码？
        self.mcts_network = mcts_network
        # 使用make_recurrent_fn_world_model创建递归函数
        self.recurrent_fn = make_recurrent_fn_world_model(
            self.mac, 
            self.batch_size, 
            self.mcts_network,
            self.temperature,
            self.n_agents + self.n_ally,
            self.k
        )
        self.replay_buffer_mcts = ReplayBuffer(
            self.episode_limit//self.c_step + 1,
            # 这个batch_size同时也需要是simulation的batch_size才行
            self.batch_size,
            self.c_step,
            use_real_data=True
        )
    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0
        self.current_skill_index = None
        self.wm_hidden_states = None
        self.replay_buffer_mcts = ReplayBuffer(
            self.episode_limit//self.c_step + 1,
            self.batch_size,
            self.c_step,
            use_real_data=True
        )
        # 存储MCTS搜索数据的临时变量
        self.current_mcts_data = None
        self.last_skill_selection_t = 0
        self.accumulated_reward = 0
        self.last_observation = None
        self.last_state = None
        self.wm_hidden_states = self.mac.init_hidden_wm(batch_size=self.batch_size, task=self.task)

    def run(self, test_mode=False, nolog=False, pretrain=False):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size, task=self.task)
        
        while not terminated:
            # 获取当前状态和观测
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()],
            }
            self.batch.update(pre_transition_data, ts=self.t)
            
            current_state = self.batch["state"][:, self.t]
            current_obs = self.batch["obs"][:, self.t]
            
            if self.t % self.c_step == 0:
                # 存储上一个周期的MCTS数据（如果有）
                if self.current_mcts_data is not None and self.t > 0:
                    policy_output, experienced_thresholds, advantages, root_policy_hidden_state, root_critic_hidden_state = self.current_mcts_data
                    self.replay_buffer_mcts.push(
                        policy_output, 
                        experienced_thresholds, 
                        advantages, 
                        root_policy_hidden_state, 
                        root_critic_hidden_state,
                        real_r=self.accumulated_reward,
                        real_next_obs=current_obs,
                        real_next_state=current_state,
                        real_done=np.zeros(self.batch_size, dtype=bool)  # 中间步骤不是终止状态
                    )
                    self.accumulated_reward = 0
                
                # 使用MCTS选择一个skill
                mcts_results = self._select_skill_with_mcts()
                self.current_skill_index = mcts_results[0]
                self.current_mcts_data = mcts_results[1:]
                # self.current_skill = self.mac.get_skill(self.current_skill_index)
                self.last_skill_selection_t = self.t
                self.last_observation = current_obs
                self.last_state = current_state
            
            # 使用当前选择的skill来选择动作. 已经内置好了将skill_index转变为emb，这里不需要转了
            actions = self.mac.forward_action_skill(
                self.batch,
                t=self.t,
                skill_index=self.current_skill_index,
                task=self.task,
                test_mode=test_mode,
            )
            
            # 选择动作并与环境交互
            chosen_actions = self.action_selector.select_action(
                actions,
                self.batch["avail_actions"][:, self.t],
                t_env=self.t_env,
                test_mode=test_mode
            )
            
            reward, terminated, env_info = self.env.step(chosen_actions[0])
            episode_return += reward
            self.accumulated_reward = self.args.gamma*self.accumulated_reward + reward  # 累积当前skill周期的奖励
            
            post_transition_data = {
                "actions": chosen_actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

            # 更新环境交互总步数（仅在非测试模式下）
            if not test_mode:
                self.t_env += 1

        # Episode结束，存储最后一个周期的MCTS数据（如果有）
        if self.current_mcts_data is not None:
            last_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()],
            }
            self.batch.update(last_data, ts=self.t)
            
            final_state = self.batch["state"][:, self.t]
            final_obs = self.batch["obs"][:, self.t]
            
            policy_output, experienced_thresholds, advantages, root_policy_hidden_state, root_critic_hidden_state = self.current_mcts_data
            self.replay_buffer_mcts.push(
                policy_output, 
                experienced_thresholds, 
                advantages, 
                root_policy_hidden_state, 
                root_critic_hidden_state,
                real_r=self.accumulated_reward,
                real_next_obs=final_obs,
                real_next_state=final_state,
                real_done=np.ones(self.batch_size, dtype=bool)  # 结束状态
            )
        
        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()],
        }
        self.batch.update(last_data, ts=self.t)

        # 最后一个状态的动作选择
        actions = self.mac.forward_action_skill(
            self.batch,
            t=self.t,
            skill_index=self.current_skill_index,
            task=self.task,
            test_mode=test_mode,
        )
        
        chosen_actions = self.action_selector.select_action(
            actions,
            self.batch["avail_actions"][:, self.t],
            t_env=self.t_env,
            test_mode=test_mode
        )
        self.batch.update({"actions": chosen_actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = f"{'pretrain/' if pretrain else ''}{self.task}/{'test_' if test_mode else ''}"
        cur_stats.update(
            {
                k: cur_stats.get(k, 0) + env_info.get(k, 0)
                for k in set(cur_stats) | set(env_info)
            }
        )
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        cur_returns.append(episode_return)

        if not nolog:
            if test_mode and len(self.test_returns) == self.args.test_nepisode:
                self._log(cur_returns, cur_stats, log_prefix)
            elif (
                not test_mode
                and self.t_env - self.log_train_stats_t >= self.args.runner_log_interval
            ):
                self._log(cur_returns, cur_stats, log_prefix)
                if "offline" not in self.args.run_file:
                    if hasattr(self.mac.action_selector, "epsilon"):
                        self.logger.log_stat(
                            f"{self.task}/epsilon",
                            self.mac.action_selector.epsilon,
                            self.t_env,
                        )
                self.log_train_stats_t = self.t_env

        return self.batch, self.replay_buffer_mcts

    def _select_skill_with_mcts(self):
        """
        使用MCTS选择一个技能
        
        Returns:
            tuple: 包含以下元素的元组:
                - skill_index: 选择的技能索引
                - policy_output: MCTS搜索输出的策略
                - experienced_thresholds: 经验阈值
                - advantages: 优势值
                - root_policy_hidden_state: 策略网络的隐藏状态
                - root_critic_hidden_state: 评论家网络的隐藏状态
        """
        # batch_obs = self.batch["obs"][:, self.t]
        state_inputs = self.batch["state"][:, self.t]
        bs = self.batch.batch_size
        # 获取上一步的动作
        batch_last_action = None
        if self.t > 0:
            batch_last_action = self.batch["actions_onehot"][:, self.t - 1]
        
        # 获取初始状态的embedding
        embedding = state_inputs.reshape(bs, -1)
        # observation = batch_obs.reshape(self.batch_size, self.n_agents, -1)
        # TODO: 这里要区分一下是不是real data，只有不是的时候才需要处理. 不过暂时都处理了
        # TODO: 输出的shape是什么意思？能不能直接用来做embedding？
        obs_inputs = self.mac.preprocess_obs(self.batch, self.t, self.task)
        # 初始化根节点
        root, experienced_thresholds, root_policy_hidden_state, root_critic_hidden_state = initialize_root(
            self.mcts_network, 
            embedding, 
            obs_inputs, 
            self.k,
            # 将wm_hidden_states传入，然后随着每一步的更新而更新
            self.wm_hidden_states
        )
        
        # 运行MCTS搜索
        policy_output, timing_stats, advantages = mctx.gumbel_muzero_policy(
            params=(),
            rng_key=np.random.RandomState(),
            root=root,
            recurrent_fn=self.recurrent_fn,
            num_simulations=self.num_simulations,
            max_num_considered_actions=self.max_num_considered_actions,
            max_depth=None,
            qtransform=functools.partial(
                mctx.qtransform_completed_by_mix_value,
                use_mixed_value=self.use_mixed_value,
            ),
        )
        # TODO: 这里需要看一看hidden_states_wm以及action怎么索引到，随后更新self.wm_hidden_states
        # new_hidden_states_wm = 
        return policy_output.action, policy_output, experienced_thresholds, advantages, root_policy_hidden_state, root_critic_hidden_state

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(
                    prefix + k + "_mean", v / stats["n_episodes"], self.t_env
                )
        stats.clear()

    def evaluate(self):
        self.reset()
        import os

        os.environ["SDL_VIDEODRIVER"] = "dummy"
        render_images = [self.env.render()]

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size, task=self.task)

        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()],
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(
                self.batch,
                t_ep=self.t,
                t_env=self.t_env,
                task=self.task,
                test_mode=True,
            )

            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

            render_images.append(self.env.render())

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()],
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(
            self.batch, t_ep=self.t, t_env=self.t_env, task=self.task, test_mode=True
        )

        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats
        cur_returns = self.test_returns
        log_prefix = f"{self.task}/eval_"
        cur_stats.update(
            {
                k: cur_stats.get(k, 0) + env_info.get(k, 0)
                for k in set(cur_stats) | set(env_info)
            }
        )
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        cur_returns.append(episode_return)

        if len(self.test_returns) == self.args.test_nepisode:
            self._log(cur_returns, cur_stats, log_prefix)

        return render_images
