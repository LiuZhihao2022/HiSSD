import gym
import torch
import numpy as np
from typing import Tuple, Any
import mctx
from mctx._src.network import PolicyValueNetwork, compute_prior_from_qvalues
from mctx._src.simple_env import SimpleEnv, MultiAgentSimpleEnv
from mctx._src.utils import stochastic_top_k_sampling, compute_offline_value_weight
from mctx._src.network import PolicyRNN

class EnvironmentWrapper:
    def __init__(self, env_name: str):
        self.env = gym.make(env_name)
        self.reset()

    def reset(self):
        state = self.env.reset()
        return state

    def step(self, action: int, state: np.ndarray) -> Tuple[np.ndarray, float, bool, Any]:
        # if self.env.done:
        #     raise ValueError("Environment is done. Please reset the environment.")
        # TODO:怎么将一个jnp action转化到一个需要实时值的环境中
        action = np.array(action)
        next_state, reward, self.done, info = self.env.step(action, state)
        return next_state, reward, self.done, info

def make_recurrent_fn_gym(env_name: str, batch_size: int, model: PolicyValueNetwork, temperature: float):
    envs = [SimpleEnv() for _ in range(batch_size)]

    def recurrent_fn(params, rng_key, action, embedding, policy_hidden_state, critic_hidden_state):
        del params, rng_key
        
        next_states = []
        rewards = []
        discounts = []

        for env, act, emb in zip(envs, action, embedding):
            next_state, reward, done, _ = env.step(act, emb)
            discount = 1.0
            next_states.append(next_state)
            rewards.append(reward)
            discounts.append(discount)

        next_states = np.array(next_states)
        rewards = np.array(rewards)
        discounts = np.array(discounts)

        # 使用新的网络预测
        prior_logits, value, new_policy_hidden_state, new_critic_hidden_state = model(next_states, policy_hidden_state, critic_hidden_state)

        recurrent_fn_output = mctx.RecurrentFnOutput(
            reward=rewards,
            discount=discounts,
            prior_logits=prior_logits,
            value=value
        )
        return recurrent_fn_output, next_states, new_policy_hidden_state, new_critic_hidden_state

    return recurrent_fn

def make_multiagent_recurrent_fn_gym(envs, batch_size: int, model: PolicyRNN, temperature: float, num_agents: int, k: int):
    # envs = [MultiAgentSimpleEnv(num_agents) for _ in range(batch_size)]

    def recurrent_fn(params, rng_key, actions, embeddings, policy_hidden_states, critic_hidden_states):
        del params, rng_key

        next_states = []
        observations = []
        rewards = []
        discounts = []
        observations = []
        actions = np.array(actions)
        for env, action_batch, embedding_batch in zip(envs, actions, embeddings):
            
            observation, next_state, reward, done, _ = env.step(action_batch, embedding_batch)
            discount = 1  # Assuming discount factor of 1.0 for all agents

            next_states.append(next_state)
            rewards.append(reward)
            discounts.append(discount)
            observations.append(observation)

        next_states = np.array(next_states)
        rewards = np.array(rewards).flatten()
        discounts = np.array(discounts).flatten()
        observations = np.array(observations)

        # TODO:此处需要对stochastic_top_k_sampling进行并行化的处理
        batched_sampled_queues, new_policy_hidden_states = stochastic_top_k_sampling(num_agents, model, observations, policy_hidden_states, model.num_actions, k)
        new_policy_hidden_states = new_policy_hidden_states.detach().cpu().numpy()
        # 提取partial actions和prior logits
        sampled_actions = [[action for action, _, _ in batch] for batch in batched_sampled_queues]
        prior_logits = [[log_prob for _, log_prob, _ in batch] for batch in batched_sampled_queues]
        sampled_actions = np.array(sampled_actions)
        prior_logits = np.array(prior_logits)
        # 使用model计算选取动作的value
        value, new_critic_hidden_states = model.predict_value(next_states, critic_hidden_states)
        value = value.detach().cpu().numpy().flatten()
        new_critic_hidden_states = new_critic_hidden_states.detach().cpu().numpy()


        # 使用新的网络预测
        # prior_logits, value, new_policy_hidden_states, new_critic_hidden_states = model(observations, policy_hidden_states, critic_hidden_states)

        recurrent_fn_output = mctx.RecurrentFnOutput(
            reward=rewards,
            discount=discounts,
            prior_logits=prior_logits,
            value=value,
            policy_hidden_states=new_policy_hidden_states,
            critic_hidden_states=new_critic_hidden_states,
            sampled_actions=sampled_actions,
        )
        return recurrent_fn_output, next_states, observations

    return recurrent_fn

class WorldModelEnv:
    """使用智能体的世界模型来模拟环境"""
    def __init__(self, mac, batch_obs, batch_last_action, wm_hidden_states, task):
        self.mac = mac
        self.current_obs = batch_obs
        self.last_action = batch_last_action
        self.hidden_states = wm_hidden_states
        self.task = task
        self.total_agents = mac.get_total_agents(task)
        
    def step(self, skill_index, embedding):
        # 使用world model预测下一步
        next_obs, next_state, reward, value, new_hidden_states = self.mac.world_model_predict(
            self.current_obs, 
            self.last_action, 
            skill_index,
            self.hidden_states,
            self.task
        )
        
        self.current_obs = next_obs
        self.last_action = None  # 重置last_action，因为我们不知道下一步的实际动作
        self.hidden_states = new_hidden_states
        
        # 返回next_state作为observation给MCTS使用, 总是不预测是否结束，只一直进行下去
        return next_obs, next_state, reward.item(), False, {}

# 注意：这个num_agents, 是实际需要选择skill的agents。在目前的sc2实现中，是agent+enemy
def make_recurrent_fn_world_model(mac, batch_size: int, model, temperature: float, num_agents: int, k: int, 
                                 offline_value_start: float = 1.0, 
                                 offline_value_end: float = 0.2, 
                                 offline_value_anneal_time: int = 500000):
    """
    创建使用世界模型的递归函数，用于MCTS搜索
    
    Args:
        mac: 多智能体控制器
        batch_size: 批大小
        model: 策略值网络模型
        temperature: 温度参数，用于探索
        num_agents: 智能体数量
        k: 采样的动作数量
        offline_value_start: offline value权重的初始值
        offline_value_end: offline value权重的最终值
        offline_value_anneal_time: 权重从初始值衰减到最终值所需的步骤数
        
    Returns:
        recurrent_fn: MCTS使用的递归函数
    """
    
    def recurrent_fn(params, rng_key, actions, observations, states, policy_hidden_states, critic_hidden_states, wm_hidden_states, task, t_step):
        del params, rng_key
        
        actions = np.array(actions)
        bs = observations.shape[0]

        next_observations ,next_states, rewards, wm_value, new_wm_hidden_states = mac.world_model_predict(observations, states, actions, wm_hidden_states, task)
        c_step =  mac.c_step
        gamma = mac.main_args.gamma
        # 计算折扣因子
        discounts = np.ones(bs)*(gamma**c_step)  # Assuming discount factor of 1.0 for all agents

        
        # 使用stochastic_top_k_sampling获取动作
        batched_sampled_queues, new_policy_hidden_states = stochastic_top_k_sampling(
            num_agents, model, next_observations, policy_hidden_states, model.num_actions, k
        )
        new_policy_hidden_states = new_policy_hidden_states.detach().cpu().numpy()
        
        # 提取采样的动作和先验概率
        sampled_actions = [[action for action, _, _ in batch] for batch in batched_sampled_queues]
        prior_logits = [[log_prob for _, log_prob, _ in batch] for batch in batched_sampled_queues]
        sampled_actions = np.array(sampled_actions)
        prior_logits = np.array(prior_logits)
        
        # 预测价值
        value, new_critic_hidden_states = model.predict_value(next_states, critic_hidden_states)
        value = value.detach().cpu().numpy().flatten()
        
        # 使用offline_value_weight来加权平均
        # TODO: 暂时先完全使用wm中的预测值来作为value，因为wm现在也要持续进行训练了
        offline_value_weight = compute_offline_value_weight(t_step, offline_value_start, offline_value_end, offline_value_anneal_time)
        value = (1 - offline_value_weight) * value + offline_value_weight * wm_value
        
        new_critic_hidden_states = new_critic_hidden_states.detach().cpu().numpy()
        
        # 返回MCTS所需的输出
        recurrent_fn_output = mctx.RecurrentFnOutput(
            reward=rewards,
            discount=discounts,
            prior_logits=prior_logits,
            value=value,
            policy_hidden_states=new_policy_hidden_states,
            critic_hidden_states=new_critic_hidden_states,
            sampled_actions=sampled_actions,
            wm_hidden_states=new_wm_hidden_states,
        )
        return recurrent_fn_output, next_states, next_observations
        
    return recurrent_fn

def make_recurrent_fn_real_simulator(mac, batch_size: int, model, temperature: float, num_agents: int, k: int, 
                                   offline_value_start: float = 1.0, 
                                   offline_value_end: float = 0.2, 
                                   offline_value_anneal_time: int = 500000,
                                   real_env=None):
    """
    创建使用真实模拟器的递归函数，用于MCTS搜索
    
    Args:
        mac: 多智能体控制器
        batch_size: 批大小
        model: 策略值网络模型
        temperature: 温度参数，用于探索
        num_agents: 智能体数量
        k: 采样的动作数量
        offline_value_start: offline value权重的初始值
        offline_value_end: offline value权重的最终值
        offline_value_anneal_time: 权重从初始值衰减到最终值所需的步骤数
        real_env: 真实环境实例，需要有real_simulator_predict方法
        
    Returns:
        recurrent_fn: MCTS使用的递归函数
    """
    
    def recurrent_fn(params, rng_key, actions, observations, states, policy_hidden_states, critic_hidden_states, wm_hidden_states, task, t_step):
        del params, rng_key
        
        actions = np.array(actions)
        bs = observations.shape[0]

        # 使用真实模拟器替代世界模型预测
        if real_env is not None and hasattr(real_env, 'real_simulator_predict'):
            try:
                # 调用真实模拟器进行预测
                # real_simulator_predict期望的输入格式: (observations, states, actions)
                # 返回格式: (next_obs, next_states, rewards, info)
                next_observations, next_states, rewards, info = real_env.real_simulator_predict(
                    observations, states, actions
                )
                
                # 确保rewards是正确的格式
                if isinstance(rewards, (int, float)):
                    rewards = np.full(bs, rewards)
                elif isinstance(rewards, np.ndarray) and rewards.ndim == 0:
                    rewards = np.full(bs, rewards.item())
                else:
                    rewards = np.array(rewards).flatten()
                
                # 计算价值预测
                import torch as th
                import torch.nn.functional as F
                device = next(mac.agent.parameters()).device
                # 将numpy数组转换为torch tensor
                next_observations_tensor = th.from_numpy(next_observations).float().to(device)
                next_states_tensor = th.from_numpy(next_states).float().to(device)
                actions_tensor = th.from_numpy(actions).long().to(device)
                
                # 构建完整输入（包括one-hot skill表示）
                one_hot_code = F.one_hot(actions_tensor, num_classes=model.num_actions).float()
                n_agents = next_observations_tensor.shape[1]
                agent_id = th.eye(n_agents, device=device).unsqueeze(0).expand(bs, -1, -1)
                
                # 分离raw observations (假设前obs_size维是原始观察)
                obs_size = real_env.obs_size if hasattr(real_env, 'obs_size') else next_observations_tensor.shape[-1] // 2
                next_raw_obs = next_observations_tensor[:, :, :obs_size]
                
                next_obs_input = [next_raw_obs, one_hot_code, agent_id]
                next_obs_input = th.cat([x.reshape(bs * n_agents, -1) for x in next_obs_input], dim=1)
                # Move to the same device as the network
                next_obs_input = next_obs_input.to(device)
                
                # 预测价值
                if hasattr(mac.agent, 'forward_value'):
                    # 获取或初始化hidden state
                    if wm_hidden_states is not None and len(wm_hidden_states.shape) >= 3:
                        # 假设wm_hidden_states包含value hidden state
                        hidden_state_value = wm_hidden_states[:, -1] if wm_hidden_states.shape[1] > 1 else wm_hidden_states[:, 0]
                    else:
                        # 使用默认的hidden state
                        hidden_state_value = mac.agent._get_hidden_states().get('value', None)
                    if hidden_state_value is not None:
                        hidden_state_value = th.from_numpy(hidden_state_value).float().to(next_obs_input.device)
                    else:
                        hidden_state_value = th.zeros(bs, mac.agent.args.rnn_hidden_dim, device=next_obs_input.device)
                    value_pred_pre, hidden_state_value = mac.agent.forward_value(
                        next_obs_input, hidden_state_value
                    )
                    
                    if hasattr(mac, 'mixer') and mac.mixer is not None:
                        value_pred = mac.mixer(
                            value_pred_pre.reshape(bs, 1, n_agents, -1), 
                            next_states_tensor.reshape(bs, 1, next_states_tensor.shape[-1])
                        )
                        value_pred = value_pred.reshape(bs,)
                    else:
                        value_pred = value_pred_pre.reshape(bs, n_agents).sum(dim=1)
                    
                    wm_value = value_pred.detach().cpu().numpy()
                    
                    # 更新wm_hidden_states
                    if wm_hidden_states is not None:
                        # 更新value部分的hidden state
                        new_wm_hidden_states = wm_hidden_states.copy()
                        if len(new_wm_hidden_states.shape) >= 3 and new_wm_hidden_states.shape[1] > 1:
                            new_wm_hidden_states[:, -1] = hidden_state_value.reshape(bs, n_agents, -1).detach().cpu().numpy()   
                    else:
                        new_wm_hidden_states = hidden_state_value.reshape(bs, n_agents, -1).detach().cpu().numpy()
                else:
                    wm_value = np.zeros(bs)
                    new_wm_hidden_states = wm_hidden_states
                
            except Exception as e:
                print(f"Warning: Real simulator prediction failed: {e}")
                raise e
                # Fallback: 保持当前状态不变
                next_observations = observations
                next_states = states
                rewards = np.zeros(bs)
                wm_value = np.zeros(bs)
                new_wm_hidden_states = wm_hidden_states
        else:
            print("Warning: Real environment not available or doesn't have real_simulator_predict method")
            raise ValueError("Real environment not available or doesn't have real_simulator_predict method")
            # Fallback: 保持当前状态不变
            next_observations = observations
            next_states = states
            rewards = np.zeros(bs)
            wm_value = np.zeros(bs)
            new_wm_hidden_states = wm_hidden_states

        c_step = mac.c_step
        gamma = mac.args.gamma
        # 计算折扣因子
        discounts = np.ones(bs) * (gamma**c_step)

        # 使用stochastic_top_k_sampling获取动作 (与world_model版本完全一致)
        batched_sampled_queues, new_policy_hidden_states = stochastic_top_k_sampling(
            num_agents, model, next_observations, policy_hidden_states, model.num_actions, k
        )
        new_policy_hidden_states = new_policy_hidden_states.detach().cpu().numpy()
        
        # 提取采样的动作和先验概率 (与world_model版本完全一致)
        sampled_actions = [[action for action, _, _ in batch] for batch in batched_sampled_queues]
        prior_logits = [[log_prob for _, log_prob, _ in batch] for batch in batched_sampled_queues]
        sampled_actions = np.array(sampled_actions)
        prior_logits = np.array(prior_logits)
        
        # 预测价值 (与world_model版本完全一致)
        value, new_critic_hidden_states = model.predict_value(next_states, critic_hidden_states)
        value = value.detach().cpu().numpy().flatten()
        
        # 使用offline_value_weight来加权平均
        offline_value_weight = compute_offline_value_weight(t_step, offline_value_start, offline_value_end, offline_value_anneal_time)
        value = (1 - offline_value_weight) * value + offline_value_weight * wm_value
        
        new_critic_hidden_states = new_critic_hidden_states.detach().cpu().numpy()
        
        # 返回MCTS所需的输出 (与world_model版本完全一致)
        recurrent_fn_output = mctx.RecurrentFnOutput(
            reward=rewards,
            discount=discounts,
            prior_logits=prior_logits,
            value=value,
            policy_hidden_states=new_policy_hidden_states,
            critic_hidden_states=new_critic_hidden_states,
            sampled_actions=sampled_actions,
            wm_hidden_states=new_wm_hidden_states,
        )
        return recurrent_fn_output, next_states, next_observations
        
    return recurrent_fn
