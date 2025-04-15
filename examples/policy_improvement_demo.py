# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A demonstration of the policy improvement by planning with Gumbel."""
import sys
import os
import time
# sys.path.append('/Users/liuzhihao/Downloads/code/mctx-main')
sys.path.append('/home/liuzhihao/HiSSD')
# print(sys.path)
import functools
from typing import Tuple, Optional
import chex
from absl import app
from absl import flags
import numpy as np
import torch
import gym
import mctx
from mctx._src.network import PolicyValueNetwork, PolicyRNN, ReplayBuffer, compute_prior_from_qvalues
from mctx._src.optimizer_wrapper import ValueOptimizerWrapper
from mctx._src.simple_env import SimpleEnv
from mctx._src.recurrent_fn import make_recurrent_fn_gym, make_recurrent_fn_world_model, make_multiagent_recurrent_fn_gym
# from mctx._src.utils import convert_tree_to_graph, stochastic_top_k_sampling
from mctx._src.utils import stochastic_top_k_sampling

FLAGS = flags.FLAGS
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("batch_size", 16, "Batch size.")
flags.DEFINE_integer("num_simulations",32, "Number of simulations.")
flags.DEFINE_integer("max_num_considered_actions", 16,
                     "The maximum number of actions expanded at the root.")
flags.DEFINE_integer("num_runs", 1000, "Number of runs on random data.")
# ma-gumbel-muzero是一个on-policy算法，因为在计算经验阈值概率的时候需要用已经保存的pertubed value和当前网络计算出来的comb_logits，所以不能存很长的buffer
flags.DEFINE_integer("replay_buffer_capacity", 1, "Capacity of the replay buffer.")
flags.DEFINE_integer("target_update_interval", 20, "Interval for updating the target network.")
flags.DEFINE_integer("state_inputs_dim", 5, "Dimension of the embedding.")
flags.DEFINE_integer("observation_dim", 1, "Dimension of the observation.")
flags.DEFINE_boolean("use_pmap", False, "Whether to use pmap for parallel training.")

# 采样的都是一颗tree，不过tree里有很多root
flags.DEFINE_integer("sample_batch_size", 1, "Batch size for sampling from the replay buffer.")
flags.DEFINE_float("lr", 5e-3, "Learning rate.")
flags.DEFINE_string("env_type", "gym", "Type of environment: 'gym' or 'world_model'.")
flags.DEFINE_integer("num_actions", 4, "Number of actions in the environment.")
flags.DEFINE_integer("max_depth", None, "The maximum search depth.")
flags.DEFINE_string("output_dir", "./output_fig",
                    "The output directory for the visualization.")
# 温度越高，概率分布越平均，因为取得是温度的倒数
flags.DEFINE_float("temperature_start", 1., "The starting temperature for the Gumbel-softmax.")
flags.DEFINE_float("temperature_end", 0.33, "The ending temperature for the Gumbel-softmax.")
flags.DEFINE_boolean("use_mixed_value", False, "Whether to use mixed value for the completed Q-value.")
flags.DEFINE_integer("k", 10, "Number of actions to sample in stochastic_top_k_sampling.")
flags.DEFINE_integer("num_agents", 5, "Number of agents in the environment.")
@chex.dataclass(frozen=True)
class DemoOutput:
  prior_policy_value: chex.Array
  prior_policy_action_value: chex.Array
  selected_action_value: chex.Array
  action_weights_policy_value: chex.Array

def initialize_root(network: PolicyRNN, state, observation, k: int, num_agents=FLAGS.num_agents, num_actions=FLAGS.num_actions, policy_hidden_states = None, critic_hidden_states = None, wm_hidden_states = None) -> mctx.RootFnOutput:
    """
    Initializes the root node for the MCTS (Monte Carlo Tree Search) process.

    Args:
        network (PolicyRNN): The policy-value network used to predict policy and value.
        state_inputs: The embedding of the current state.
        observation: The observation of the current state.
        temperature: The temperature parameter for stochastic sampling.
        k (int): The number of top actions to sample.
        policy_hidden_states: The hidden states of the policy network. Defaults to None.
        critic_hidden_states: The hidden states of the critic network. Defaults to None.

    Returns:
        mctx.RootFnOutput: The initialized root node containing prior logits, value, embedding, observation,
                           new policy hidden states, new critic hidden states, and sampled actions.
    """
    # 初始化hidden states
    batch_size = state.shape[0]
    if wm_hidden_states is not None and not isinstance(wm_hidden_states, np.ndarray):
        if isinstance(wm_hidden_states, torch.Tensor):
            wm_hidden_states = wm_hidden_states.detach().cpu().numpy()
        else:
            wm_hidden_states = np.array(wm_hidden_states)
    if policy_hidden_states is None or critic_hidden_states is None:
        # policy_hidden_states, critic_hidden_states = network.init_hidden(batch_size= batch_size)
        policy_hidden_states, critic_hidden_states = network.get_hidden_states()

    # 使用stochastic_top_k_sampling选取动作
    batched_sampled_queues_with_reference, new_policy_hidden_states = stochastic_top_k_sampling(
        num_agents, network, observation, policy_hidden_states, num_actions, k+1
    )
    batched_sampled_queues = [batch[:-1] for batch in batched_sampled_queues_with_reference]
    experienced_thresholds = [batch[-1][2] for batch in batched_sampled_queues_with_reference]

    new_policy_hidden_states = new_policy_hidden_states.detach().cpu().numpy()
    # 提取partial actions和prior logits
    sampled_actions = [[action for action, _, _ in batch] for batch in batched_sampled_queues]
    prior_logits = [[log_prob for _, log_prob, _ in batch] for batch in batched_sampled_queues]
    sampled_actions = np.array(sampled_actions)
    prior_logits = np.array(prior_logits)
    # 使用model计算选取动作的value
    value, new_critic_hidden_states = network.predict_value(state, critic_hidden_states)
    value = value.detach().cpu().numpy().flatten()
    new_critic_hidden_states = new_critic_hidden_states.detach().cpu().numpy()
    root = mctx.RootFnOutput(
        prior_logits=prior_logits,
        value=value,
        embedding=state,
        observation=observation,
        new_policy_hidden_states=new_policy_hidden_states,
        new_critic_hidden_states=new_critic_hidden_states,
        new_wm_hidden_states=wm_hidden_states,
        sampled_actions=sampled_actions
    )
    return root, experienced_thresholds, policy_hidden_states, critic_hidden_states

def _run_demo(rng, network: PolicyRNN, recurrent_fn, temperature) -> Tuple[np.random.RandomState, DemoOutput, ValueOptimizerWrapper]:
    batch_size = FLAGS.batch_size
    rng, value_rng, search_rng = np.random.RandomState(), np.random.RandomState(), np.random.RandomState()

    # 创建batch的初始状态embedding
    embedding = np.random.randint(0, 10, size=(batch_size, FLAGS.state_inputs_dim))
    # TODO:这里还要改成多个agent的情况
    observation = embedding.reshape(batch_size, FLAGS.num_agents, FLAGS.observation_dim)

    # 初始化树根
    root, experienced_thresholds, root_policy_hidden_state, root_critic_hidden_state = initialize_root(network, embedding, observation, FLAGS.k)
    # Running the search
    policy_output, timing_stats, advantages = mctx.gumbel_muzero_policy(
        params=(),
        rng_key=search_rng,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=FLAGS.num_simulations,
        task=task,
        max_num_considered_actions=FLAGS.max_num_considered_actions,
        max_depth=FLAGS.max_depth,
        qtransform=functools.partial(
            mctx.qtransform_completed_by_mix_value,
            use_mixed_value=FLAGS.use_mixed_value,),
    )
    return rng, policy_output, timing_stats, experienced_thresholds, advantages, root_policy_hidden_state, root_critic_hidden_state

def main(_):
    # 显示CUDA是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    rng = np.random.RandomState(FLAGS.seed)
    replay_buffer = ReplayBuffer(FLAGS.replay_buffer_capacity)
    
    # 创建环境并生成recurrent_fn
    # network = PolicyValueNetwork(FLAGS.state_inputs_dim, FLAGS.num_actions)
    network = PolicyRNN(FLAGS.observation_dim, FLAGS.state_inputs_dim, FLAGS.num_actions, FLAGS.num_agents)
    print(f"Neural network is using device: {network.device}")
    # network.initialize(lr=FLAGS.lr)
    jitted_run_demo = _run_demo
    
    # 添加时间统计变量
    total_stats = {
        'search_time': 0.0,
        'simulate_time': 0.0,
        'expand_time': 0.0,
        'backward_time': 0.0,
        'network_time': 0.0,
        'total_runs': 0,
        'train_time': 0.0
    }
    if FLAGS.env_type == "gym":
        env_name = "SimpleEnv-v0"
        # gym.envs.registration.register(id=env_name, entry_point=SimpleEnv)
        from mctx._src.simple_env import MultiAgentSimpleEnv
        envs = [MultiAgentSimpleEnv(FLAGS.num_agents) for _ in range(FLAGS.batch_size)]

    elif FLAGS.env_type == "world_model":
        raise NotImplementedError("World model is not implemented yet.")
    else:
        raise ValueError(f"Unsupported environment type: {FLAGS.env_type}")
    
    for i in range(FLAGS.num_runs):
        # 需要吧更新后的网络传入，以给每一个状态确定先验以及value
        tempreature = FLAGS.temperature_start - (FLAGS.temperature_start - FLAGS.temperature_end) * i / FLAGS.num_runs
        recurrent_fn = make_multiagent_recurrent_fn_gym(envs, FLAGS.batch_size, network, tempreature, FLAGS.num_agents, FLAGS.k)
        # 记录整体搜索时间
        search_start = time.time()
        # rng, output, policy_output, timing_stats = jitted_run_demo(
        #     rng, network, recurrent_fn, tempreature
        # )
        rng, policy_output, timing_stats, experienced_thresholds, advantages, root_policy_hidden_state, root_critic_hidden_state = jitted_run_demo(
            rng, network, recurrent_fn, tempreature
        )
        del recurrent_fn
        search_time = time.time() - search_start
        
        # 收集各阶段时间统计
        total_stats['search_time'] += search_time
        total_stats['simulate_time'] += timing_stats.get('simulate_time', 0.0)
        total_stats['expand_time'] += timing_stats.get('expand_time', 0.0)
        total_stats['backward_time'] += timing_stats.get('backward_time', 0.0)
        total_stats['network_time'] += timing_stats.get('network_time', 0.0)
        total_stats['total_runs'] += 1

        replay_buffer.push(policy_output, experienced_thresholds, advantages, root_policy_hidden_state, root_critic_hidden_state)

        train_start = time.time()
        if len(replay_buffer) >= FLAGS.sample_batch_size:
            batch = replay_buffer.sample(FLAGS.sample_batch_size)
            
            network.train_network(batch=batch)
            
        if i % FLAGS.target_update_interval == 0:
            network.update_target_network()
        total_stats['train_time'] += time.time() - train_start
        
        # if i == 0:
        #    initial_action_value = output.selected_action_value

        # action_value_improvement_relative_to_initial = (
        #     output.selected_action_value - initial_action_value
        # )
        # weights_value_improvement_relative_to_initial = (
        #     output.action_weights_policy_value - output.prior_policy_value
        # )

        if i % 1 == 0:
            print(f"Run {i}")
            # print("action value improvement relative to initial:         %.3f (min=%.3f)" %
            #       (action_value_improvement_relative_to_initial.mean(), action_value_improvement_relative_to_initial.min()))
            # print("action_weights value improvement relative to initial: %.3f (min=%.3f)" %
            #       (weights_value_improvement_relative_to_initial.mean(), weights_value_improvement_relative_to_initial.min()))
            # 添加平均时间统计输出
            runs = total_stats['total_runs']
            print("\nAverage timing statistics:")
            print(f"Search time: {total_stats['search_time']/runs:.3f}s")
            print(f"Simulate time: {total_stats['simulate_time']/runs:.3f}s")
            print(f"Expand time: {total_stats['expand_time']/runs:.3f}s")
            print(f"Backward time: {total_stats['backward_time']/runs:.3f}s")
            print(f"Network time: {total_stats['network_time']/runs:.3f}s")
            print(f"Train time: {total_stats['train_time']/runs:.3f}s")
            # print("Time statistics:")
            # print(f"Search time: {search_time:.3f}s")
            # print(f"Simulate time: {timing_stats['simulate_time']:.3f}s")
            # print(f"Expand time: {timing_stats['expand_time']:.3f}s")
            # print(f"Backward time: {timing_stats['backward_time']:.3f}s")
            print("----------------------------------------")
        # TODO:这里打印的prior和画图画出来的对不上？
        # 画图里面还进行了一次softmax。prior_logits到底是概率还是logits？
        # 是logits。那么就不应该转换为概率输入进去
        # if i % 50 == 0:
        #     # test_embedding = np.arange(0, 10).reshape(-1, 1)
        #     # test_prior_logits, test_qvalues = network.predict(test_embedding)
        #     # print("Q-values for test embedding:")
        #     # print(test_qvalues)
        #     # print("Prior policy for test embedding:")
        #     # print(compute_prior_from_qvalues(test_qvalues, tempreature, max_min_transform=True))
        #     graph = convert_tree_to_graph(policy_output.search_tree, tree_action_as_label = True, embedding_as_label=True, temperature=tempreature)
            
        #     print("Saving tree diagram to:", os.path.join(FLAGS.output_dir, f"tree_{i}.png"))
        #     graph.draw(os.path.join(FLAGS.output_dir, f"tree_{i}.png"), prog="dot")

if __name__ == "__main__":
  app.run(main)
