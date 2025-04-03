import gym
from gym import spaces
import numpy as np

class SimpleEnv(gym.Env):
    def __init__(self):
        super(SimpleEnv, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=10, shape=(1,), dtype=np.float32)
        state = None
        self.done = np.zeros(self.num_agents, dtype=bool)

    def reset(self):
        state = np.array([5.0])
        self.done = False
        return state

    def step(self, action, state):
        # if self.done:
        #     raise ValueError("Environment is done. Please reset the environment.")
        
        if action == 0:
            state -= 1
        elif action == 1:
            state += 1
        elif action == 2:
            state *= 2
        elif action == 3:
            state /= 2

        reward = -np.abs(state - 5).item()
        # state = np.where(state < 0, 0, state)
        # state = np.where(state > 10, 10, state)
        self.done = state <= 0 or state >= 10
        return state, reward, self.done, {}
        
    def render(self, mode='human'):
        pass

    def close(self):
        pass

import gym
from gym import spaces
import numpy as np

class MultiAgentSimpleEnv(gym.Env):
    def __init__(self, num_agents=3):
        super(MultiAgentSimpleEnv, self).__init__()
        self.num_agents = num_agents
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=10, shape=(1,), dtype=np.float32
        )  # 每个智能体的观察为单一维度
        self.state_space = spaces.Box(
            low=0, high=10, shape=(self.num_agents,), dtype=np.float32
        )  # 全局状态为所有智能体状态的拼接
        self.states = np.full((self.num_agents,), 5.0)
        self.done = np.zeros(self.num_agents, dtype=bool)

    def reset(self, initial_states=None):
        """
        Resets the environment to an initial state.

        Args:
            initial_states (np.ndarray, optional): The initial states for each agent. Shape should be (num_agents,).
                                                   If None, default initial state is used.

        Returns:
            np.ndarray: Observations for each agent.
        """
        if initial_states is not None:
            assert initial_states.shape == (self.num_agents,), "initial_states shape should be (num_agents,)"
            self.states = initial_states
        else:
            self.states = np.full((self.num_agents,), 5.0)

        self.done = False
        return self._get_observations(), 

    def step(self, actions, state=None):
        """
        Advances the environment by one step.

        Args:
            actions (np.ndarray): Actions for each agent. Shape should be (num_agents,).
            state (np.ndarray, optional): State for each agent to start from in this step. Shape should be (num_agents,).

        Returns:
            tuple: Tuple containing:
                - observations (np.ndarray): Observations for each agent.
                - rewards (float): The reward for all agents.
                - done (np.ndarray): Done flags for each agent.
                - info (dict): Additional information.
        """
        if state is not None:
            assert state.shape == (self.num_agents,), "state shape should be (self.num_agents,)"
            self.states = state

        next_states = self.states.copy()

        for i in range(self.num_agents):
            # if self.done[i]:
            #     continue

            action = actions[i]

            # Action effects
            try:
                if action == 0:  # Decrease
                    next_states[i] -= 1
                elif action == 1:  # Increase
                    next_states[i] += 1
                elif action == 2:  # Double
                    next_states[i] *= 2
                elif action == 3:  # Halve
                    next_states[i] /= 2
            except TypeError as e:
                if action == 0:  # Decrease
                    next_states = next_states.at[i].set(next_states[i] - 1)
                elif action == 1:  # Increase
                    next_states = next_states.at[i].set(next_states[i] + 1)
                elif action == 2:  # Double
                    next_states = next_states.at[i].set(next_states[i] * 2)
                elif action == 3:  # Halve
                    next_states = next_states.at[i].set(next_states[i] / 2)
            

            # next_states[i] = state[i]

            # Done condition
            # TODO: done 暂时不使用
            # self.done[i] = next_states[i] <= 0 or next_states[i] >= 10

        self.states = next_states
        global_state_mean = np.mean(next_states)  # Encourage cooperation toward a common goal
        reward = -np.abs(global_state_mean - 5).item()  # Reward is higher when all agents cooperate to center

        return self._get_observations(), self._get_state(), reward, self.done.copy(), {}

    def _get_observations(self):
        """
        Returns individual observations for each agent.
        Each agent observes only its own state.

        Returns:
            np.ndarray: Observations for each agent, shape (num_agents, 1).
        """
        return self.states[:, None]
    
    def _get_state(self):
        """
        Returns the global state of the environment.

        Returns:
            np.ndarray: The global state of the environment, shape (num_agents,).
        """
        return self.states

    def render(self, mode='human'):
        print(f"States: {self.states}, Done: {self.done}")

    def close(self):
        pass