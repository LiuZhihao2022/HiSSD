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
"""A JAX implementation of batched MCTS."""
import functools
from typing import Any, NamedTuple, Optional, Tuple, TypeVar
import numpy as np
import chex
from mctx._src import action_selection
from mctx._src import base
from mctx._src import tree as tree_lib
import jax
import jax.numpy as jnp
import multiprocessing as mp
import time
import torch
Tree = tree_lib.Tree
T = TypeVar("T")


def search(
    params: base.Params,
    rng_key: np.random.RandomState,
    *,
    root: base.RootFnOutput,
    recurrent_fn: base.RecurrentFn,
    root_action_selection_fn: base.RootActionSelectionFn,
    interior_action_selection_fn: base.InteriorActionSelectionFn,
    num_simulations: int,
    max_depth: Optional[int] = None,
    invalid_actions: Optional[np.ndarray] = None,
    extra_data: Any = None,
    loop_fn: base.LoopFn = None,
    batch_size: int = 32) -> Tree:
  """Performs a full search and returns sampled actions.

  In the shape descriptions, `B` denotes the batch dimension.

  Args:
    params: params to be forwarded to root and recurrent functions.
    rng_key: random number generator state, the key is consumed.
    root: a `(prior_logits, value, embedding)` `RootFnOutput`. The
      `prior_logits` are from a policy network. The shapes are
      `([B, num_actions], [B], [B, ...])`, respectively.
    recurrent_fn: a callable to be called on the leaf nodes and unvisited
      actions retrieved by the simulation step, which takes as args
      `(params, rng_key, action, embedding)` and returns a `RecurrentFnOutput`
      and the new state embedding. The `rng_key` argument is consumed.
    root_action_selection_fn: function used to select an action at the root.
    interior_action_selection_fn: function used to select an action during
      simulation.
    num_simulations: the number of simulations.
    max_depth: maximum search tree depth allowed during simulation, defined as
      the number of edges from the root to a leaf node.
    invalid_actions: a mask with invalid actions at the root. In the
      mask, invalid actions have ones, and valid actions have zeros.
      Shape `[B, num_actions]`.
    extra_data: extra data passed to `tree.extra_data`. Shape `[B, ...]`.
    loop_fn: Function used to run the simulations. It may be required to pass
      hk.fori_loop if using this function inside a Haiku module.
    target_network_params: Optional parameters for the target network.
    batch_size: Batch size for training the value network.

  Returns:
    `SearchResults` containing outcomes of the search, e.g. `visit_counts`
    `[B, num_actions]`.
  """
  action_selection_fn = action_selection.switching_action_selection_wrapper(
      root_action_selection_fn=root_action_selection_fn,
      interior_action_selection_fn=interior_action_selection_fn
  )

  batch_size = root.value.shape[0]
  batch_range = np.arange(batch_size)
  if max_depth is None:
    max_depth = num_simulations
  if invalid_actions is None:
    invalid_actions = np.zeros_like(root.prior_logits)

  # 初始化时间统计
  timing_stats = {
      'simulate_time': 0.0,
      'expand_time': 0.0,
      'backward_time': 0.0
  }

  def body_fun(sim, loop_state):
    rng_key, tree = loop_state
    
    # Simulate timing
    sim_start = time.time()
    parent_index, action = simulate(tree, action_selection_fn ,max_depth)
    timing_stats['simulate_time'] += time.time() - sim_start
    
    next_node_index = tree.children_index[batch_range, parent_index, action]
    next_node_index = np.where(next_node_index == Tree.UNVISITED, sim + 1, next_node_index)
    
    # Expand timing
    expand_start = time.time()
    tree = expand(params, tree, recurrent_fn, parent_index, action, next_node_index)
    timing_stats['expand_time'] += time.time() - expand_start
    
    # Backward timing
    backward_start = time.time()
    tree = backward(tree, next_node_index)
    timing_stats['backward_time'] += time.time() - backward_start
    
    loop_state = rng_key, tree
    return loop_state

  tree = instantiate_tree_from_root(root, num_simulations,
                                    root_invalid_actions=invalid_actions,
                                    extra_data=extra_data)
  for sim in range(num_simulations):
    rng_key, tree = body_fun(sim, (rng_key, tree))

  return tree, timing_stats

class _SimulationState(NamedTuple):
  """The state for the simulation while loop."""
  node_index: int
  action: int
  next_node_index: int
  depth: int
  is_continuing: bool
  
@functools.partial(jax.vmap, in_axes=[0, None, None], out_axes=0)
def simulate(
    tree: Tree,
    action_selection_fn: base.InteriorActionSelectionFn,
    max_depth: int) -> Tuple[chex.Array, chex.Array]:
  """Traverses the tree until reaching an unvisited action or `max_depth`.

  Each simulation starts from the root and keeps selecting actions traversing
  the tree until a leaf or `max_depth` is reached.

  Args:
    rng_key: random number generator state, the key is consumed.
    tree: _unbatched_ MCTS tree state.
    action_selection_fn: function used to select an action during simulation.
    max_depth: maximum search tree depth allowed during simulation.

  Returns:
    `(parent_index, action)` tuple, where `parent_index` is the index of the
    node reached at the end of the simulation, and the `action` is the action to
    evaluate from the `parent_index`.
  """
  def cond_fun(state):
    return state.is_continuing

  def body_fun(state):
    # Preparing the next simulation state.
    node_index = state.next_node_index
    action = action_selection_fn(None, tree, node_index,
                                 state.depth)
    next_node_index = tree.children_index[node_index, action]
    # The returned action will be visited.
    depth = state.depth + 1
    is_before_depth_cutoff = depth < max_depth
    is_visited = next_node_index != Tree.UNVISITED
    is_continuing = jnp.logical_and(is_visited, is_before_depth_cutoff)
    return _SimulationState(  # pytype: disable=wrong-arg-types  # jax-types
        node_index=node_index,
        action=action,
        next_node_index=next_node_index,
        depth=depth,
        is_continuing=is_continuing)

  node_index = jnp.array(Tree.ROOT_INDEX, dtype=jnp.int32)
  depth = jnp.zeros((), dtype=tree.children_prior_logits.dtype)
  # pytype: disable=wrong-arg-types  # jnp-type
  initial_state = _SimulationState(
      node_index=tree.NO_PARENT,
      action=tree.NO_PARENT,
      next_node_index=node_index,
      depth=depth,
      is_continuing=jnp.array(True))
  # pytype: enable=wrong-arg-types
  end_state = jax.lax.while_loop(cond_fun, body_fun, initial_state)

  # Returning a node with a selected action.
  # The action can be already visited, if the max_depth is reached.
  return end_state.node_index, end_state.action

def expand(
    params: np.ndarray,
    tree: Tree[T],
    recurrent_fn: base.RecurrentFn,
    parent_index: np.ndarray,
    action: np.ndarray,
    next_node_index: np.ndarray) -> Tree[T]:
  """Create and evaluate child nodes from given nodes and unvisited actions.

  Args:
    params: params to be forwarded to recurrent function.
    rng_key: random number generator state.
    tree: the MCTS tree state to update.
    hidden_state_tree: the hidden state tree to maintain hidden states.
    recurrent_fn: a callable to be called on the leaf nodes and unvisited
      actions retrieved by the simulation step, which takes as args
      `(params, rng_key, action, embedding)` and returns a `RecurrentFnOutput`
      and the new state embedding. The `rng_key` argument is consumed.
    parent_index: the index of the parent node, from which the action will be
      expanded. Shape `[B]`.
    action: the action to expand. Shape `[B]`.
    next_node_index: the index of the newly expanded node. This can be the index
      of an existing node, if `max_depth` is reached. Shape `[B]`.

  Returns:
    tree: updated MCTS tree state.
  """
  batch_size = tree_lib.infer_batch_size(tree)
  batch_range = np.arange(batch_size)
  parent_index = np.array(parent_index)
  action = np.array(action)
  corresponding_joint_action = jax.vmap(extract_actions, in_axes=(0, 0, 0))(tree.sampled_actions, action, parent_index)
  chex.assert_shape([parent_index, action, next_node_index], (batch_size,))
  embedding = jax.tree_util.tree_map(
      lambda x: x[batch_range, parent_index], tree.embeddings)

  observation = jax.tree_util.tree_map(
      lambda x: x[batch_range, :, parent_index], tree.observations)

  policy_hidden_states = jax.tree_util.tree_map(
      lambda x: x[batch_range, parent_index], tree.policy_hidden_states)
  critic_hidden_states = jax.tree_util.tree_map(
      lambda x: x[batch_range, parent_index], tree.critic_hidden_states)
  
  embedding = np.array(embedding)
  observation = np.array(observation)
  policy_hidden_states = np.array(policy_hidden_states)
  critic_hidden_states = np.array(critic_hidden_states)
  step, next_embedding, next_observation = recurrent_fn(params, None, corresponding_joint_action, embedding, policy_hidden_states, critic_hidden_states)
  # TODO: 这里解包后还需要改改
  chex.assert_shape(step.prior_logits, [batch_size, tree.num_actions])
  chex.assert_shape(step.reward, [batch_size])
  chex.assert_shape(step.discount, [batch_size])
  chex.assert_shape(step.value, [batch_size])
  tree = update_tree_node(
      tree, next_node_index, step.prior_logits, step.value, next_embedding, next_observation, step.policy_hidden_states, step.critic_hidden_states, step.sampled_actions)

  return tree.replace(
      children_index=batch_update(
          tree.children_index, next_node_index, parent_index, action),
      children_rewards=batch_update(
          tree.children_rewards, step.reward, parent_index, action),
      children_discounts=batch_update(
          tree.children_discounts, step.discount, parent_index, action),
      parents=batch_update(tree.parents, parent_index, next_node_index),
      action_from_parent=batch_update(
          tree.action_from_parent, action, next_node_index))

@jax.vmap
def backward(
    tree: Tree[T],
    leaf_index: chex.Numeric) -> Tree[T]:
  """Goes up and updates the tree until all nodes reached the root.

  Args:
    tree: the MCTS tree state to update, without the batch size.
    leaf_index: the node index from which to do the backward.

  Returns:
    Updated MCTS tree state.
  """

  def cond_fun(loop_state):
    _, _, index = loop_state
    return index != Tree.ROOT_INDEX

  def body_fun(loop_state):
    # Here we update the value of our parent, so we start by reversing.
    # leaf_value就是当前这个节点的value
    tree, leaf_value, index = loop_state
    parent = tree.parents[index]
    # 无论选去哪个动作，parent节点的计数都会+1，因为其肯定是其父节点选出来的
    count = tree.node_visits[parent]
    action = tree.action_from_parent[index]
    # 更新网络的时候，使用对应动作的children rewards就可以？
    reward = tree.children_rewards[parent, action]
    # 对于叶节点，leaf_value默认为0，所以更新后就等于reward; 对于中间节点，则为累积未来回报
    # 此时，leaf_value被更新为parent value的一次采样。用的方法是：reward + discount * leaf_value
    leaf_value = reward + tree.children_discounts[parent, action] * leaf_value
    # 注意：不能使用这个value值去训练神经网络！这个value值
    parent_value = (
        tree.node_values[parent] * count + leaf_value) / (count + 1.0)
    children_values = tree.node_values[index]
    # 只有选取的action的计数才会+1
    children_counts = tree.children_visits[parent, action] + 1

    tree = tree.replace(
        node_values=update(tree.node_values, parent_value, parent),
        node_visits=update(tree.node_visits, count + 1, parent),
        children_values=update(
            tree.children_values, children_values, parent, action),
        children_visits=update(
            tree.children_visits, children_counts, parent, action))

    return tree, leaf_value, parent

  leaf_index = jnp.asarray(leaf_index, dtype=jnp.int32)
  loop_state = (tree, tree.node_values[leaf_index], leaf_index)
  tree, _, _ = jax.lax.while_loop(cond_fun, body_fun, loop_state)

  return tree

def update(x, vals, *indices):
  # x[indices] = vals
  # return x
  return x.at[indices].set(vals)

batch_update = jax.vmap(update)

def extract_actions(tree_sampled_actions, action, parent_index):
    return tree_sampled_actions[parent_index, action]

# def batch_update(xs, vals, indices):
#     """
#     Batch updates multiple inputs along axis=0.

#     Args:
#         xs: array-like, shape (batch_size, ...)
#         vals: array-like, shape (batch_size, num_updates)
#         indices: array-like, shape (batch_size, ...)

#     Returns:
#         Updated array `xs` with the same shape.
#     """
#     batch_size = xs.shape[0]
#     batch_indices = np.arange(batch_size)[:, None]  # Batch index for each update
#     xs[batch_indices, indices] = vals
#     return xs

def update_tree_node(
    tree: Tree[T],
    node_index: np.ndarray,
    prior_logits: np.ndarray,
    value: np.ndarray,
    embedding: np.ndarray,
    observations: np.ndarray,
    new_policy_hidden_states: np.ndarray,
    new_critic_hidden_states: np.ndarray,
    sampled_actions: np.ndarray) -> Tree[T]:
  """Updates the tree at node index."""
  batch_size = tree_lib.infer_batch_size(tree)
  batch_range = np.arange(batch_size)
  chex.assert_shape(prior_logits, (batch_size, tree.num_actions))
  new_visit = tree.node_visits[batch_range, node_index] + 1
  updates = dict(
      children_prior_logits=batch_update(
          tree.children_prior_logits, prior_logits, node_index),
      raw_values=batch_update(
          tree.raw_values, value, node_index),
      node_values=batch_update(
          tree.node_values, value, node_index),
      node_visits=batch_update(
          tree.node_visits, new_visit, node_index),
      embeddings=jax.tree_util.tree_map(
          lambda t, s: batch_update(t, s, node_index),
          tree.embeddings, embedding),
      observations=jax.tree_util.tree_map(
          lambda t, s: batch_update(t, s, node_index),
          tree.observations, observations),
      policy_hidden_states=batch_update(tree.policy_hidden_states, new_policy_hidden_states, node_index),
      critic_hidden_states=batch_update(tree.critic_hidden_states, new_critic_hidden_states, node_index),
      sampled_actions=batch_update(tree.sampled_actions, sampled_actions, node_index)
  )

  return tree.replace(**updates)

def instantiate_tree_from_root(
    root: base.RootFnOutput,
    num_simulations: int,
    root_invalid_actions: np.ndarray,
    extra_data: Any) -> Tree:
  """Initializes tree state at search root."""
  chex.assert_rank(root.prior_logits, 2)
  # num_actions其实就是k
  batch_size, num_actions = root.prior_logits.shape
  num_agents = root.new_policy_hidden_states.shape[-2]
  chex.assert_shape(root.value, [batch_size])
  num_nodes = num_simulations + 1
  data_dtype = root.value.dtype
  batch_node = (batch_size, num_nodes)
  batch_node_action = (batch_size, num_nodes, num_actions)

  def _zeros(x):
    return np.zeros(batch_node + x.shape[1:], dtype=x.dtype)

  tree = Tree(
      node_visits=np.zeros(batch_node, dtype=np.int32),
      raw_values=np.zeros(batch_node, dtype=data_dtype),
      node_values=np.zeros(batch_node, dtype=data_dtype),
      parents=np.full(batch_node, Tree.NO_PARENT, dtype=np.int32),
      action_from_parent=np.full(
          batch_node, Tree.NO_PARENT, dtype=np.int32),
      children_index=np.full(
          batch_node_action, Tree.UNVISITED, dtype=np.int32),
      children_prior_logits=np.zeros(
          batch_node_action, dtype=root.prior_logits.dtype),
      children_values=np.zeros(batch_node_action, dtype=data_dtype),
      children_visits=np.zeros(batch_node_action, dtype=np.int32),
      children_rewards=np.zeros(batch_node_action, dtype=data_dtype),
      children_discounts=np.zeros(batch_node_action, dtype=data_dtype),
      embeddings=jax.tree_util.tree_map(_zeros, root.embedding),
      observations=jax.tree_util.tree_map(_zeros, root.observation),
      root_invalid_actions=root_invalid_actions,
      extra_data=extra_data,
      # 最后一个维度才和agent的数目有关
      sampled_actions=np.zeros((batch_size, num_nodes, num_actions, num_agents), dtype=np.int32),
      policy_hidden_states=np.zeros((batch_size, num_nodes, num_agents, root.new_policy_hidden_states.shape[-1]), dtype=np.float32),
      critic_hidden_states=np.zeros((batch_size, num_nodes, root.new_critic_hidden_states.shape[-1]), dtype=np.float32)
  )

  root_index = np.full([batch_size], Tree.ROOT_INDEX)
  tree = update_tree_node(
      tree, root_index, root.prior_logits, root.value, root.embedding, root.observation, root.new_policy_hidden_states, root.new_critic_hidden_states, root.sampled_actions)
  return tree
