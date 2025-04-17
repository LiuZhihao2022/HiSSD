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
"""A data structure used to hold / inspect search data for a batch of inputs."""

from __future__ import annotations
from typing import Any, ClassVar, Generic, TypeVar
import chex
import numpy as np
import jax
import torch

T = TypeVar("T")

@chex.dataclass(frozen=True)
class Tree(Generic[T]):
  """State of a search tree.

  The `Tree` dataclass is used to hold and inspect search data for a batch of
  inputs. In the fields below `B` denotes the batch dimension, `N` represents
  the number of nodes in the tree, and `num_actions` is the number of discrete
  actions.

  node_visits: `[B, N]` the visit counts for each node.
  raw_values: `[B, N]` the raw value for each node. Maybe estimated by a neural network or other models.
  node_values: `[B, N]` the cumulative search value for each node.
  parents: `[B, N]` the node index for the parents for each node.
  action_from_parent: `[B, N]` action to take from the parent to reach each
    node.
  children_index: `[B, N, num_actions]` the node index of the children for each
    action.
  children_prior_logits: `[B, N, num_actions]` the action prior logits of each
    node.
  children_visits: `[B, N, num_actions]` the visit counts for children for
    each action.
  children_rewards: `[B, N, num_actions]` the immediate reward for each action.
  children_discounts: `[B, N, num_actions]` the discount between the
    `children_rewards` and the `children_values`.
  children_values: `[B, N, num_actions]` the value of the next node after the
    action.
  embeddings: `[B, N, ...]` the state embeddings of each node.
  observation: `[B, N, num_agents, ...]` the observations of each node.
  root_invalid_actions: `[B, num_actions]` a mask with invalid actions at the
    root. In the mask, invalid actions have ones, and valid actions have zeros.
  extra_data: `[B, ...]` extra data passed to the search.
  sampled_actions: np.ndarray  # [B, k]
  policy_hidden_states: torch.FloatTensor  # [B, N, num_agents, hidden_dim]
  critic_hidden_states: torch.FloatTensor  # [B, N, hidden_dim]
  """
  node_visits: np.ndarray  # [B, N]
  raw_values: np.ndarray  # [B, N]
  node_values: np.ndarray  # [B, N]
  parents: np.ndarray  # [B, N]
  action_from_parent: np.ndarray  # [B, N]
  children_index: np.ndarray  # [B, N, num_actions]
  children_prior_logits: np.ndarray  # [B, N, num_actions]
  children_visits: np.ndarray  # [B, N, num_actions]
  children_rewards: np.ndarray  # [B, N, num_actions]
  children_discounts: np.ndarray  # [B, N, num_actions]
  children_values: np.ndarray  # [B, N, num_actions]
  embeddings: Any  # [B, N, ...]
  observations: np.ndarray  # [B, N, num_agents, ...]
  sampled_actions: np.ndarray  # [B, N, k]
  root_invalid_actions: np.ndarray  # [B, num_actions]
  extra_data: T  # [B, ...]
  sampled_actions: np.ndarray  # [B, k]
  policy_hidden_states: torch.FloatTensor  # [B, N, hidden_dim]
  critic_hidden_states: torch.FloatTensor  # [B, N, hidden_dim]
  # TODO: check，wm里面是一个list而不是单纯的tensor
  wm_hidden_states: torch.FloatTensor  # [B, N, hidden_dim]
  # The following attributes are class variables (and should not be set on
  # Tree instances).
  ROOT_INDEX: ClassVar[int] = 0
  NO_PARENT: ClassVar[int] = -1
  UNVISITED: ClassVar[int] = -1

  @property
  def num_actions(self):
    return self.children_index.shape[-1]

  @property
  def num_simulations(self):
    return self.node_visits.shape[-1] - 1

  def qvalues(self, indices):
    """Compute q-values for any node indices in the tree."""
    # if np.asarray(indices).shape:
    #   return np.vectorize(_unbatched_qvalues)(self, indices)
    # else:
    return _unbatched_qvalues(self, indices)
  # def qvalues(self, indices):
  #   """Compute q-values for any node indices in the tree."""
  #   # pytype: disable=wrong-arg-types  # jnp-type
  #   if np.array(indices).shape:
  #     return jax.vmap(_unbatched_qvalues)(self, indices)
  #   else:
  #     return _unbatched_qvalues(self, indices)
    
  def summary(self) -> SearchSummary:
    """Extract summary statistics for the root node."""
    value = self.node_values[:, Tree.ROOT_INDEX]
    batch_size, = value.shape
    root_indices = np.full((batch_size,), Tree.ROOT_INDEX)
    qvalues = self.qvalues(root_indices)
    visit_counts = self.children_visits[:, Tree.ROOT_INDEX].astype(value.dtype)
    total_counts = np.sum(visit_counts, axis=-1, keepdims=True)
    # 这里计算了访问概率。但是这个概率并不是gumble muzero中的改善策略的访问概率呀？
    visit_probs = visit_counts / np.maximum(total_counts, 1)
    visit_probs = np.where(total_counts > 0, visit_probs, 1 / self.num_actions)
    return SearchSummary(
        visit_counts=visit_counts,
        visit_probs=visit_probs,
        value=value,
        qvalues=qvalues)

  def get_single_batch(self, batch_idx: int) -> "Tree":
    """返回只包含单个batch数据的新Tree对象。"""
    # 只取batch_idx对应的那一行/切片
    def _slice(x):
      if isinstance(x, np.ndarray) or (hasattr(x, 'shape') and hasattr(x, '__getitem__')):
        if x.shape[0] == self.node_values.shape[0]:
          return x[batch_idx:batch_idx+1]
        return x
      return x
    return Tree(
      node_visits=_slice(self.node_visits),
      raw_values=_slice(self.raw_values),
      node_values=_slice(self.node_values),
      parents=_slice(self.parents),
      action_from_parent=_slice(self.action_from_parent),
      children_index=_slice(self.children_index),
      children_prior_logits=_slice(self.children_prior_logits),
      children_visits=_slice(self.children_visits),
      children_rewards=_slice(self.children_rewards),
      children_discounts=_slice(self.children_discounts),
      children_values=_slice(self.children_values),
      embeddings=_slice(self.embeddings),
      observations=_slice(self.observations),
      sampled_actions=_slice(self.sampled_actions),
      root_invalid_actions=_slice(self.root_invalid_actions),
      extra_data=_slice(self.extra_data),
      policy_hidden_states=_slice(self.policy_hidden_states),
      critic_hidden_states=_slice(self.critic_hidden_states),
      wm_hidden_states=_slice(self.wm_hidden_states),
    )


def infer_batch_size(tree: Tree) -> int:
  """Recovers batch size from `Tree` data structure."""
  if tree.node_values.ndim != 2:
    raise ValueError("Input tree is not batched.")
  return tree.node_values.shape[0]

@chex.dataclass(frozen=True)
class SearchSummary:
  """Stats from MCTS search."""
  visit_counts: np.ndarray
  visit_probs: np.ndarray
  value: np.ndarray
  qvalues: np.ndarray


def _unbatched_qvalues(tree: Tree, index: tuple) -> int:
  # chex.assert_rank(tree.children_discounts, 2)
  return (
      tree.children_rewards[index]
      + tree.children_discounts[index] * tree.children_values[index]
  )
