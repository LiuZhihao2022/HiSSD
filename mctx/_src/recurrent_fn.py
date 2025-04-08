import gym
import torch
import numpy as np
from typing import Tuple, Any
import mctx
from mctx._src.network import PolicyValueNetwork, compute_prior_from_qvalues
from mctx._src.simple_env import SimpleEnv, MultiAgentSimpleEnv
from mctx._src.utils import stochastic_top_k_sampling
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
def make_recurrent_fn_world_model(mac, batch_size: int, model, temperature: float, num_agents: int, k: int):
    """
    创建使用世界模型的递归函数，用于MCTS搜索
    
    Args:
        envs: 世界模型环境列表
        batch_size: 批大小
        model: 策略值网络模型
        temperature: 温度参数，用于探索
        num_agents: 智能体数量
        k: 采样的动作数量
        
    Returns:
        recurrent_fn: MCTS使用的递归函数
    """
    
    def recurrent_fn(params, rng_key, actions, last_actions, embeddings, policy_hidden_states, critic_hidden_states, wm_hidden_states, task):
        del params, rng_key
        
        actions = np.array(actions)
        bs = embeddings.shape[0]

        observations ,next_states, rewards, values, new_wm_hidden_states = mac.world_model_predict(embeddings, last_actions, actions, wm_hidden_states, task)
        discounts = np.ones(bs)  # Assuming discount factor of 1.0 for all agents

        
        # 使用stochastic_top_k_sampling获取动作
        batched_sampled_queues, new_policy_hidden_states = stochastic_top_k_sampling(
            num_agents, model, observations, policy_hidden_states, model.num_actions, k
        )
        new_policy_hidden_states = new_policy_hidden_states.detach().cpu().numpy()
        new_wm_hidden_states = new_wm_hidden_states.detach().cpu().numpy()
        # 提取采样的动作和先验概率
        sampled_actions = [[action for action, _, _ in batch] for batch in batched_sampled_queues]
        prior_logits = [[log_prob for _, log_prob, _ in batch] for batch in batched_sampled_queues]
        sampled_actions = np.array(sampled_actions)
        prior_logits = np.array(prior_logits)
        
        # 预测价值
        value, new_critic_hidden_states = model.predict_value(next_states, critic_hidden_states)
        # TODO: 暂时将value和values进行平均。后面可能更改为随着时间的推移进行更新
        # TODO: check shape
        value = (value + values)/2
        value = value.detach().cpu().numpy().flatten()
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
        return recurrent_fn_output, next_states, observations
        
    return recurrent_fn
