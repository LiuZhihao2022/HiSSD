import torch as th
import numpy as np
 #from .base_decomposer import BaseDecomposer # Assuming BaseDecomposer is in .base_decomposer

class MPEDecomposer:
    def __init__(self, task_args, main_args):
        # super().__init__(main_args)
        self.task_args = task_args
        self.main_args = main_args

        self.n_agents = self.task_args.n_agents
        # In simple_spread, num_landmarks = num_agents.
        # Allowing for scenarios where n_landmarks might be specified differently.
        self.n_landmarks = getattr(self.task_args, 'n_landmarks', self.n_agents)

        # Standard MPE observation structure:
        # obs: [self_vel(2), self_pos(2), other_agent_rel_pos((N-1)*2), landmark_rel_pos(L*2)]
        self._own_feats_dim_raw = 4  # self_vel (2) + self_pos (2)
        self._ally_feats_dim_raw = 2  # relative position of other agents (2)
        self._landmark_feats_dim_raw = 2 # relative position of landmarks (2)

        # This is the observation dim for a single agent, coming from env_info
        self.obs_dim = self.task_args.obs_shape 
        
        # Validate the received obs_dim against the assumed structure
        expected_obs_dim = self._own_feats_dim_raw + \
                           (self.n_agents - 1) * self._ally_feats_dim_raw + \
                           self.n_landmarks * self._landmark_feats_dim_raw
        
        if self.obs_dim != expected_obs_dim:
            print(f"Warning: MPEDecomposer obs_dim mismatch in task {self.task_args.env_args.get('scenario_name', 'unknown_mpe')}. "
                  f"Expected {expected_obs_dim} based on (own_vel(2)+own_pos(2) + (n_agents-1)*other_pos(2) + n_landmarks*landmark_pos(2)), "
                  f"but got {self.obs_dim} from task_args.obs_shape. Check MPE observation structure and MPEEnvWrapper.")
            # Potentially raise an error or use a more flexible decomposition if this is a common issue.
            # For now, we proceed, but decomposition might be incorrect.

        self.state_dim = self.task_args.state_shape
        self.entity_embed_dim = getattr(self.main_args, "entity_embed_dim", 0) # Ensure it exists

        # For compatibility with components like MTAttnMixer,
        # we map landmarks to "enemies" and other agents to "allies".
        self.n_enemies = self.n_landmarks # Landmarks are treated as "enemies"
        self.n_allies = self.n_agents - 1  # Other agents

    @property
    def own_feats_dim(self):
        """Dimension of own features after decomposition."""
        return self._own_feats_dim_raw

    @property
    def ally_feats_dim(self):
        """Dimension of ally features after decomposition."""
        return self._ally_feats_dim_raw

    @property
    def enemy_feats_dim(self):
        """Dimension of enemy (landmark) features after decomposition."""
        return self._landmark_feats_dim_raw

    @property
    def entity_shapes(self):
        """
        Returns a dictionary of raw feature dimensions for different entity types.
        Used by HISSDAgent or other modules.
        """
        return {
            "self": self._own_feats_dim_raw,
            "ally": self._ally_feats_dim_raw,
            "enemy": self._landmark_feats_dim_raw, # Landmarks mapped to enemies
            "landmark": self._landmark_feats_dim_raw # Explicit landmark features
        }

    def get_obs_dim(self):
        """Returns the raw observation dimension for a single agent."""
        return self.obs_dim

    def get_state_dim(self):
        """Returns the state dimension."""
        return self.state_dim

    def decompose_obs(self, obs_batch, agent_id=None):
        """
        Decomposes the observation batch.
        obs_batch shape: (..., obs_dim) where ... can be (batch_size, n_agents) or (batch_size) etc.
        Returns: own_features, enemy_features (landmarks), ally_features
        """
        original_shape = obs_batch.shape
        obs_dim_from_batch = original_shape[-1]

        if obs_dim_from_batch != self.obs_dim:
            # This might happen if the obs_dim check in __init__ passed due to some default
            # but the actual data has a different dimension.
            raise ValueError(f"Observation dimension in batch ({obs_dim_from_batch}) "
                             f"does not match decomposer's expected obs_dim ({self.obs_dim}).")

        # Determine the shape of the leading dimensions (e.g., batch_size, n_agents)
        leading_dims = original_shape[:-1]

        # Calculate slicing points
        own_end_idx = self._own_feats_dim_raw
        ally_end_idx = own_end_idx + (self.n_agents - 1) * self._ally_feats_dim_raw
        # landmark_end_idx = ally_end_idx + self.n_landmarks * self._landmark_feats_dim_raw # Should be obs_dim

        # Slice features
        own_features = obs_batch[..., :own_end_idx]

        if self.n_allies > 0:
            ally_features_flat = obs_batch[..., own_end_idx:ally_end_idx]
            ally_features = ally_features_flat.reshape(*leading_dims, self.n_allies, self._ally_feats_dim_raw)
        else: # Handle case with only 1 agent (no allies)
            ally_features = th.empty(*leading_dims, 0, self._ally_feats_dim_raw, device=obs_batch.device)


        if self.n_landmarks > 0:
            landmark_features_flat = obs_batch[..., ally_end_idx:self.obs_dim] # Use self.obs_dim to ensure all data is captured
            landmark_features = landmark_features_flat.reshape(*leading_dims, self.n_landmarks, self._landmark_feats_dim_raw)
        else: # Handle case with no landmarks
            landmark_features = th.empty(*leading_dims, 0, self._landmark_feats_dim_raw, device=obs_batch.device)
            
        # Return in the order: own, enemy (landmarks), ally
        return own_features, landmark_features, ally_features

    def get_entity_embeddings(self, obs_batch, agent_id=None):
        # This method might be part of a more complex BaseDecomposer.
        # For now, it's a placeholder if needed by other parts of the system.
        # Typically, entity features are first decomposed, then embedded by a model.
        raise NotImplementedError("get_entity_embeddings is not implemented in MPEDecomposer. Embedding should be handled by the agent/model.")

