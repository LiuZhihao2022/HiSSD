import dataclasses
import jax.numpy as jnp

@dataclasses.dataclass
class MuzeroTrajectory:
    """A trajectory of observations, actions, rewards, policies, and values."""
    # History of actions taken by the agent. Shape (trajectory_length,).
    action_history: jnp.ndarray
    # History of rewards received by the agent. Shape (trajectory_length,).
    reward_history: jnp.ndarray
    # History of policy logits from MCTS. Shape (trajectory_length, num_actions).
    policy_history: jnp.ndarray
    # History of values from MCTS root. Shape (trajectory_length,).
    value_history: jnp.ndarray
    # History of discount factors. Shape (trajectory_length,).
    discount_history: jnp.ndarray
    # History of embeddings (hidden states). Shape (trajectory_length + 1, embedding_dim).
    # Includes the initial embedding and subsequent embeddings after each action.
    embedding_history: jnp.ndarray

    def __post_init__(self):
        # Basic validation
        num_transitions = len(self.action_history)
        if not (len(self.reward_history) == num_transitions and
                len(self.policy_history) == num_transitions and
                len(self.value_history) == num_transitions and
                len(self.discount_history) == num_transitions and
                len(self.embedding_history) == num_transitions + 1): # Embeddings for S_0 to S_T
            raise ValueError("History lengths must match for a valid trajectory.")

    def __len__(self):
        return len(self.action_history)
