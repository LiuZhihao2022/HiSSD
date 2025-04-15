from typing import Optional, Sequence
import chex
import numpy as np
from typing import List, Tuple
import torch
import jax
import jax.numpy as jnp

def compute_offline_value_weight(t: int, offline_value_start: float, offline_value_end: float, offline_value_anneal_time: int) -> float:
    """
    计算随着时间的推移而衰减的offline value权重
    
    Args:
        t: 当前的训练步骤
        offline_value_start: 初始权重值
        offline_value_end: 最终权重值
        offline_value_anneal_time: 权重从初始值衰减到最终值所需的步骤数
        
    Returns:
        float: 当前步骤的offline value权重
    """
    # 线性衰减
    frac = min(1.0, t / offline_value_anneal_time)
    return offline_value_start + frac * (offline_value_end - offline_value_start)

def convert_tree_to_graph(
    tree,
    action_labels: Optional[Sequence[str]] = None,
    tree_action_as_label: bool = False,
    batch_index: int = 0,
    embedding_as_label: bool = False,
    temperature: float = 1.0,
):
  import pygraphviz
  """Converts a search tree into a Graphviz graph.

  Args:
    tree: A `Tree` containing a batch of search data.
    action_labels: Optional labels for edges, defaults to the action index.
    batch_index: Index of the batch element to plot.

  Returns:
    A Graphviz graph representation of `tree`.
  """
  chex.assert_rank(tree.node_values, 2)
  batch_size = tree.node_values.shape[0]

  def node_to_str(node_i, reward=0, discount=1):
    node_name = f"{tree.embeddings[batch_index, node_i]}" if embedding_as_label else f"{node_i}"
    return (f"{node_name}\n"
            f"Reward: {reward:.2f}\n"
            f"Discount: {discount:.2f}\n"
            # 这个values是神经网络+MCTS模拟得到的，而visits是MCTS模拟得到的
            f"Value: {tree.node_values[batch_index, node_i]:.2f}\n"
            f"Visits: {tree.node_visits[batch_index, node_i]}\n")

  def edge_to_str(node_i, a_i, action_labels):
    # node_index = jnp.full([batch_size], node_i)
    probs = np.exp(tree.children_prior_logits[batch_index, node_i]) / np.sum(np.exp(tree.children_prior_logits[batch_index, node_i]))
    return (f"{action_labels[a_i]}\n"
            # 这个Q是用MCTS模拟得到的，而probs是用神经网络得到的，所以Q和p的计算对不上
            f"Q: {tree.qvalues((batch_index, node_i, a_i)):.2f}\n"  # pytype: disable=unsupported-operands  # always-use-return-annotations
            f"p: {probs[a_i]:.2f}\n")

  graph = pygraphviz.AGraph(directed=True)

  # Add root
  graph.add_node(0, label=node_to_str(node_i=0), color="green")
  # Add all other nodes and connect them up.
  for node_i in range(tree.num_simulations):
    if tree_action_as_label:
        action_labels = tree.sampled_actions[batch_index, node_i]
    elif action_labels is None:
      action_labels = range(tree.num_actions)
    elif len(action_labels) != tree.num_actions:
      raise ValueError(
          f"action_labels {action_labels} has the wrong number of actions "
          f"({len(action_labels)}). "
          f"Expecting {tree.num_actions}.")
    for a_i in range(tree.num_actions):
      # Index of children, or -1 if not expanded
      children_i = tree.children_index[batch_index, node_i, a_i]
      if children_i >= 0:
        graph.add_node(
            children_i,
            label=node_to_str(
                node_i=children_i,
                reward=tree.children_rewards[batch_index, node_i, a_i],
                discount=tree.children_discounts[batch_index, node_i, a_i]),
            color="red")
        graph.add_edge(node_i, children_i, label=edge_to_str(node_i, a_i, action_labels))

  return graph



def stochastic_top_k_sampling(
    n_agents: int,
    policy_rnn,
    observations: np.ndarray,
    policy_hidden_states: np.ndarray,
    max_actions: int,
    k: int,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Implements Algorithm 1: Stochastically sample top-k joint actions without replacement.

    Args:
        n_agents (int): Number of agents.
        policy_rnn (PolicyRNN): RNN model for computing action probabilities.
        observations (np.ndarray) : Initial states of shape [batch_size, num_agents, obs_dim].
        policy_hidden_states (np.ndarray) : Hidden states of shape [batch_size, num_agents, hidden_dim].
        max_actions (int): Maximum number of actions per agent.
        k (int): Number of joint actions to sample.

    Returns:
        List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]: Top-k joint actions, their log probabilities, and Gumbel values.
    """
    batch_size = observations.shape[0]
    # Initialize batched_queue with empty joint actions
    # [partial_action, log_prob, gumbel_value]
    batched_queue = [[([], 0.0, 0.0)] for _ in range(batch_size)]  # Initialize empty batched_queue
    logits, new_policy_hidden_states = policy_rnn.predict_policy(observations, policy_hidden_states)  # [batch_size, num_agents, max_actions]
    logits = logits.detach().cpu().numpy()
    # new_policy_hidden_states = new_policy_hidden_states.detach().cpu().numpy()
    for agent_idx in range(n_agents):
        for b in range(batch_size):
          queue = batched_queue[b]
          expansions = []
          for partial_action, log_prob, gumbel_value in queue:
              to_expanded = []
              Z = -float("inf")  # Track max Gumbel noise

              for action_idx in range(max_actions):
                  # Update expansions
                  new_partial_action = partial_action + [action_idx]
                  new_log_prob = log_prob + logits[b, agent_idx, action_idx]

                  # Compute Gumbel noise and perturbed value
                  perturbed_value = np.random.gumbel(loc=new_log_prob, scale=1.0, size=1)
                  Z = max(Z, perturbed_value)
                  to_expanded.append((new_partial_action, new_log_prob, perturbed_value))

              # Recalculate adjusted Gumbel values for all expansions
              for j, (partial_action, log_prob, gumbel_value) in enumerate(to_expanded):
                  adjusted_gumbel = -np.log(
                      np.exp(-gumbel_value) - np.exp(-Z) + np.exp(-gumbel_value)
                  )
                  expansions.append((partial_action, log_prob, adjusted_gumbel))

          # Sort expansions by adjusted Gumbel values and keep top-k
          expansions.sort(key=lambda x: x[2], reverse=True)
          batched_queue[b] = expansions[:k]

    return batched_queue, new_policy_hidden_states

@jax.vmap
def compute_advantage(tree, node_index):
    qvalues = tree.qvalues(node_index)
    visit_counts = tree.children_visits[node_index]
    # TODO: 这里默认使用了raw_value，而没有使用mix_value
    value = tree.raw_values[node_index]
    completed_qvalues = jnp.where(visit_counts > 0,qvalues,value)
    min_value = jnp.min(completed_qvalues, axis=-1, keepdims=False)
    max_value = jnp.max(completed_qvalues, axis=-1, keepdims=False)
    min_value = jnp.minimum(min_value, value)
    max_value = jnp.maximum(max_value, value)
    normalized_qvalues = (completed_qvalues - min_value) / jnp.maximum(max_value - min_value, 1e-8)
    normalized_value = jax.lax.cond(
        jnp.array_equal(min_value, value),
        lambda _: jnp.zeros_like(value),
        lambda _: (value - min_value) / jnp.maximum(max_value - min_value, 1e-8),
        operand=None
    )
    return normalized_qvalues - normalized_value
