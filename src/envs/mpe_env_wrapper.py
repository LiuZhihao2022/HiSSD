import gymnasium
import numpy as np
from pettingzoo.mpe import simple_spread_v3 # Using simple_spread_v3 as an example

class MPEEnvWrapper:
    def __init__(self, scenario_name="simple_spread_v3", episode_limit=100, seed=None, **kwargs):
        """
        Wrapper for PettingZoo MPE environments.
        Args:
            scenario_name (str): Name of the MPE scenario (e.g., "simple_spread_v3").
            episode_limit (int): Maximum number of steps per episode.
            seed (int, optional): Random seed for the environment.
        """
        self.scenario_name = scenario_name
        self._episode_limit = episode_limit
        self._seed = seed
        
        # Dynamically import the environment based on scenario_name
        # This is a simplified example; a more robust solution might map names to classes
        if scenario_name == "simple_spread_v3":
            self.env = simple_spread_v3.parallel_env(
                N=kwargs.get("N", 3), # Number of agents
                local_ratio=kwargs.get("local_ratio", 0.5),
                max_cycles=episode_limit,
                continuous_actions=kwargs.get("continuous_actions", False)
            )
        # Add other MPE scenarios here as needed
        # elif scenario_name == "simple_tag_v3":
        #     self.env = simple_tag_v3.parallel_env(...)
        else:
            raise ValueError(f"Unsupported MPE scenario: {scenario_name}")

        self.env.reset(seed=self._seed) # Reset to initialize agents, action_spaces, etc.
        
        self.n_agents = len(self.env.possible_agents)
        self._agent_ids = self.env.possible_agents

        # Assuming discrete actions for MPE (typically 5 actions: no-op, N, S, E, W)
        # PettingZoo action spaces are usually Discrete or Box
        first_agent_action_space = self.env.action_space(self._agent_ids[0])
        if not isinstance(first_agent_action_space, gymnasium.spaces.Discrete):
            raise ValueError("MPE environment action space is not Discrete. Continuous actions not supported by this wrapper yet.")
        self._n_actions = first_agent_action_space.n

        # Observation space
        first_agent_obs_space = self.env.observation_space(self._agent_ids[0])
        self._obs_shape = first_agent_obs_space.shape[0] # Assuming flat observation vectors

        # State shape: For MPE, often a concatenation of agent observations or a global view
        # This is a placeholder; a more specific state representation might be needed.
        self._state_shape = self._obs_shape * self.n_agents 

        self._episode_steps = 0

    def reset(self):
        """Resets the environment and returns initial observations."""
        self._episode_steps = 0
        observations_dict, infos_dict = self.env.reset(seed=self._seed)
        self._last_obs = self._dict_to_list_of_arrays(observations_dict)
        return self.get_obs(), self.get_state() # Return obs and state

    def step(self, actions):
        """
        Steps the environment with the given actions.
        Args:
            actions (list or np.ndarray): List/array of actions for each agent.
        Returns:
            tuple: (global_reward, terminated, info_dict)
        """
        if len(actions) != self.n_agents:
            raise ValueError(f"Number of actions ({len(actions)}) does not match number of agents ({self.n_agents}).")

        actions_dict = {agent_id: actions[i] for i, agent_id in enumerate(self._agent_ids)}
        
        observations_dict, rewards_dict, terminations_dict, truncations_dict, infos_dict = self.env.step(actions_dict)

        self._last_obs = self._dict_to_list_of_arrays(observations_dict)
        
        # Aggregate rewards (global reward)
        global_reward = sum(rewards_dict.values())
        
        # Check for termination (all agents terminated or truncated)
        terminated = all(terminations_dict.values()) or all(truncations_dict.values())
        
        self._episode_steps += 1
        if self._episode_steps >= self._episode_limit:
            terminated = True # Enforce episode limit

        # info can contain agent-specific details if needed
        # For now, just pass the raw infos_dict from PettingZoo
        # Add win/loss or other metrics if available and relevant
        info = {"infos_dict": infos_dict} 
        if terminated:
            # Example: Add a 'battle_won' or 'is_success' if applicable to the scenario
            # This depends on the MPE scenario specifics.
            # For simple_spread, success might be when all agents cover landmarks.
            # This logic needs to be implemented based on scenario goals.
            info["battle_won"] = False # Placeholder

        return global_reward, terminated, info

    def get_obs(self):
        """Returns the current observations for all agents as a list of arrays."""
        return self._last_obs

    def get_state(self):
        """
        Returns the global state of the environment.
        Placeholder: Concatenates all agent observations.
        """
        return np.concatenate(self._last_obs, axis=0).flatten()

    def get_avail_actions(self):
        """
        Returns a list of available actions for each agent.
        For discrete MPE, typically all actions are available.
        """
        avail_actions_list = []
        for _ in range(self.n_agents):
            avail_actions_list.append(np.ones(self._n_actions, dtype=int))
        return avail_actions_list

    def get_env_info(self):
        """Returns environment information needed by the runner and agent."""
        return {
            "state_shape": self._state_shape,
            "obs_shape": self._obs_shape,
            "n_agents": self.n_agents,
            "n_actions": self._n_actions,
            "episode_limit": self._episode_limit,
            "agent_ids": self._agent_ids
        }

    def close(self):
        """Closes the environment."""
        self.env.close()

    def _dict_to_list_of_arrays(self, data_dict):
        """Converts a dictionary of agent data to a list of numpy arrays."""
        return [np.array(data_dict[agent_id]) for agent_id in self._agent_ids]

if __name__ == '__main__':
    # Example Usage
    env_args = {"scenario_name": "simple_spread_v3", "episode_limit": 50, "N": 3, "seed": 123}
    mpe_wrapper = MPEEnvWrapper(**env_args)
    
    env_info = mpe_wrapper.get_env_info()
    print("Environment Info:", env_info)

    obs, state = mpe_wrapper.reset()
    print("Initial Obs Shape (per agent):", obs[0].shape)
    print("Initial State Shape:", state.shape)

    terminated = False
    total_reward = 0
    for _ in range(env_info["episode_limit"] + 5): # Run a bit longer to test limit
        if terminated:
            break
        actions = [np.random.randint(0, env_info["n_actions"]) for _ in range(env_info["n_agents"])]
        reward, terminated, info = mpe_wrapper.step(actions)
        total_reward += reward
        # print(f"Step: {_}, Reward: {reward}, Terminated: {terminated}")

    print("Episode Finished. Total Reward:", total_reward)
    mpe_wrapper.close()
