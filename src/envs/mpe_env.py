from pettingzoo.mpe import simple_spread_v3
import gymnasium as gym
import numpy as np
from .multiagentenv import MultiAgentEnv

class MPEWrapper(MultiAgentEnv):
    def __init__(self, key, time_limit, **kwargs):
        self.episode_limit = time_limit
        self.env_key = key
        self._env = simple_spread_v3.parallel_env(N=kwargs.get("n_agents", 3), 
                                                 local_ratio=0.5, 
                                                 max_cycles=time_limit, 
                                                 continuous_actions=False)
        self._env.reset() # Need to reset to get agent_selection, observation_space, action_space

        self.n_agents = len(self._env.possible_agents)
        self._obs = None
        self._info = None

        # Assuming discrete actions and observation spaces for MPE
        self.action_spaces = [self._env.action_space(agent) for agent in self._env.possible_agents]
        self.observation_spaces = [self._env.observation_space(agent) for agent in self._env.possible_agents]

        if not all(isinstance(space, gym.spaces.Discrete) for space in self.action_spaces):
            raise ValueError("MPEWrapper only supports Discrete action spaces.")
        if not all(isinstance(space, gym.spaces.Box) for space in self.observation_spaces):
            # MPE observations are Box, but let's be explicit
            raise ValueError("MPEWrapper expects Box observation spaces.")

        self._n_actions = self.action_spaces[0].n 
        self._obs_shape = self.observation_spaces[0].shape


    def step(self, actions):
        """ Returns reward, terminated, info """
        # PettingZoo expects actions as a dictionary {agent_id: action}
        # Our runner provides a list/numpy array of actions
        agent_actions = {}
        for i, agent in enumerate(self._env.agents): # Use current agents
            agent_actions[agent] = actions[i]

        observations, rewards, terminations, truncations, infos = self._env.step(agent_actions)
        
        self._obs = [observations[agent] for agent in self._env.possible_agents if agent in observations]
        
        # Aggregate rewards (can be customized)
        reward = sum(rewards.values()) 
        
        # terminated is True if all agents are terminated or truncated
        terminated = all(terminations.values()) or all(truncations.values())
        
        # Store info if needed, PettingZoo's info is a dict per agent
        self._info = infos 

        # Ensure observations are padded or handled if agents disappear mid-episode
        # For now, assume agents persist or their observations are handled by the policy
        
        return reward, terminated, infos # infos can be used for more detailed data

    def get_obs(self):
        """ Returns all agent observations in a list """
        # This should return observations for all *possible* agents,
        # potentially padding for agents that are done.
        # For simplicity, returning current _obs, assuming runner handles padding/masking.
        obs_list = []
        raw_obs = self._env.observe(self._env.agent_selection) # Get current agent's obs
        
        # This is a simplified way; ideally, you'd get all current obs
        # and map them to the fixed order of possible_agents.
        # The current self._obs is updated in step().
        return self._obs


    def get_obs_agent(self, agent_id_idx):
        """ Returns observation for agent_id_idx (integer index) """
        # agent_id_idx is an integer index, map to PettingZoo's agent string
        agent_str = self._env.possible_agents[agent_id_idx]
        if agent_str in self._obs: # Check if agent is currently active
             return self._obs[agent_str] # This needs to be fixed, self._obs is a list
        # Fallback or error if agent not found or inactive
        # This part needs careful handling based on how inactive agents are represented
        return np.zeros(self._obs_shape) # Placeholder for inactive agent

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return self._obs_shape[0] # Assuming Box space, return the feature dimension

    def get_state(self):
        # MPE typically doesn't have a global state. Concatenate observations.
        # This needs to be a consistent representation.
        if self._obs is None or not self._obs: # Handle initial call before first step
             # Create zero observations based on possible_agents
            flat_obs = [np.zeros(self.observation_spaces[i].shape) for i in range(self.n_agents)]
        else:
            flat_obs = [obs.flatten() for obs in self._obs] # self._obs should be a list of np arrays
        
        return np.concatenate(flat_obs, axis=0).astype(np.float32)

    def get_state_size(self):
        """ Returns the shape of the state"""
        # Sum of flattened observation sizes
        return sum(obs_space.shape[0] for obs_space in self.observation_spaces)

    def get_avail_actions(self):
        """ Returns the available actions for all agents """
        # For MPE simple_spread, all actions are typically available
        avail_actions_list = []
        for i in range(self.n_agents):
            avail_actions_list.append(self.get_avail_agent_actions(i))
        return avail_actions_list

    def get_avail_agent_actions(self, agent_id_idx):
        """ Returns the available actions for agent_id_idx """
        # For MPE simple_spread, all actions are typically available
        return [1] * self._n_actions 

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return self._n_actions

    def reset(self):
        """ Returns initial observations and states"""
        observations, infos = self._env.reset()
        self._obs = [observations[agent] for agent in self._env.possible_agents if agent in observations]
        return self.get_obs(), self.get_state() # Return obs list and state

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()

    def seed(self, seed=None):
        # PettingZoo environments are seeded at creation or reset
        # We re-create the env for a new seed to be sure.
        self._env = simple_spread_v3.parallel_env(N=self.n_agents, 
                                                 local_ratio=0.5, 
                                                 max_cycles=self.episode_limit, 
                                                 continuous_actions=False)
        self._env.reset(seed=seed)
        self.action_spaces = [self._env.action_space(agent) for agent in self._env.possible_agents]
        self.observation_spaces = [self._env.observation_space(agent) for agent in self._env.possible_agents]


    def save_replay(self):
        # Not implemented for MPE
        pass

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info

    def get_stats(self):
        return {} # Or any MPE specific stats
