from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mctx._src import tree as tree_lib
from mctx._src.optimizer_wrapper import ValueOptimizerWrapper
import copy
import jax
import torch
import jax.numpy as jnp

class PolicyValueNetwork(nn.Module):
    def __init__(self, input_size: int, num_actions: int, hidden_sizes: Tuple[int, ...] = (128, 128)):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_actions = num_actions
        
        # 共享特征提取层
        self.feature_layers = []
        current_size = input_size
        for hidden_size in hidden_sizes:
            self.feature_layers.extend([
                nn.Linear(current_size, hidden_size),
                nn.ReLU()
            ])
            current_size = hidden_size
        self.feature_network = nn.Sequential(*self.feature_layers).to(self.device)
        
        # 策略头
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_sizes[-1], hidden_sizes[-1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_sizes[-1] // 2, num_actions)
        ).to(self.device)
        
        # 价值头
        self.value_head = nn.Sequential(
            nn.Linear(hidden_sizes[-1], hidden_sizes[-1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_sizes[-1] // 2, 1)
        ).to(self.device)
        
        # 目标网络，包括feature_network和value_head
        self.target_network = nn.Sequential(
            *self.feature_layers,
            nn.Linear(hidden_sizes[-1], hidden_sizes[-1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_sizes[-1] // 2, 1)
        ).to(self.device)
        
        self.optimizer = None

    def forward(self, observation: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        observation = observation.to(self.device)
        state = state.to(self.device)
        features = self.feature_network(observation)
        policy_logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return policy_logits, value

    def predict_policy(self, observation: torch.Tensor, policy_hidden_state: torch.Tensor) -> torch.Tensor:
        observation = observation.to(self.device)
        policy_logits, _ = self.policy_network(observation, policy_hidden_state)
        return policy_logits

    def predict_value(self, state: torch.Tensor, critic_hidden_state: torch.Tensor) -> torch.Tensor:
        state = state.to(self.device)
        value, _ = self.critic_network(state, critic_hidden_state)
        return value

    def initialize(self, lr: float = 1e-3, optimizer_name: str = 'adam'):
        if optimizer_name == 'adam':
            self.optimizer = torch.optim.Adam(self.trainable_parameters(), lr=lr)
        elif optimizer_name == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(self.trainable_parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def trainable_parameters(self):
        # 返回用于训练的参数，不包括target_network的参数
        return list(self.feature_network.parameters()) + \
               list(self.policy_head.parameters()) + \
               list(self.value_head.parameters())

    def predict(self, embedding: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        embedding = ensure_tensor(embedding)
        with torch.no_grad():
            policy_logits, value = self(embedding)
            # policy_probs = F.softmax(policy_logits, dim=-1)
        return policy_logits.cpu().numpy(), value.cpu().numpy()

    def train_network(self, batch: Tuple, gamma: float = 0.99, value_loss_weight: float = 0.5, max_grad_norm: float = 10.0) -> float:
        states, actions, rewards, next_states, improved_policy_probs = prepare_batch_data(batch)
        states = torch.FloatTensor(states).to(self.device)
        # 使用最终选择的优秀动作计算Q值。即value的估计是按照最好的动作进行估计的
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        improved_policy_probs = torch.FloatTensor(improved_policy_probs).to(self.device)
        
        # 前向传播
        policy_logits, predicted_values = self(states)
        
        # 计算策略损失 (交叉熵)
        policy_loss = nn.CrossEntropyLoss()(policy_logits, improved_policy_probs)
        
        # 计算目标价值
        with torch.no_grad():
            target_values = self.target_network(next_states).squeeze(-1)
            target_values = rewards + gamma * target_values
        
        # 计算价值损失 (MSE)，使用改善后的策略分布加权
        value_loss = nn.MSELoss()(predicted_values, target_values)
        
        # 总损失 (可以调整权重)
        total_loss = policy_loss + value_loss_weight * value_loss
        
        # 更新参数
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.trainable_parameters(), max_grad_norm)
        
        self.optimizer.step()
        
        return total_loss.item()

    def update_target_network(self):
        self.target_network.load_state_dict(self.feature_network.state_dict(), strict=False)
        self.target_network[-3:].load_state_dict(self.value_head.state_dict(), strict=False)


class PolicyNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_dim=128, seed=42):
        super(PolicyNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_actions = output_shape
        self.seed = seed
        
        self.network = nn.Sequential(
            nn.Linear(input_shape, hidden_dim),
            nn.ReLU(),
            nn.GRUCell(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, output_shape)
        )
        
        torch.manual_seed(self.seed)
        nn.init.normal_(self.network[0].weight, mean=0, std=0.1)
        nn.init.normal_(self.network[3].weight, mean=0, std=0.1)

    def forward(self, inputs, hidden_state):
        # inputs = ensure_tensor(inputs)
        # hidden_state = ensure_tensor(hidden_state)
        b, a, e = inputs.shape
        inputs = inputs.view(-1, e)
        x = F.relu(self.network[0](inputs), inplace=True)
        h_in = hidden_state.reshape(-1, self.hidden_dim)
        hh = self.network[2](x, h_in)
        logits = self.network[3](hh)
        return logits.view(b, a, -1), hh.view(b, a, -1)

    def init_hidden(self, batch_size=1, num_agents=1):
        hidden_states = self.network[0].weight.new(1, self.hidden_dim).zero_()
        return hidden_states.unsqueeze(0).expand(batch_size, num_agents, -1)

    def random_init_hidden(self, seed, batch_size=1, num_agents=1):
        torch.manual_seed(seed)
        weights = torch.empty(batch_size, self.hidden_dim)
        nn.init.normal_(weights, mean=0, std=0.01)
        weights.requires_grad = True
        return weights

    def cuda(self):
        self.network.cuda()

class CriticNetwork(nn.Module):
    def __init__(self, input_shape, hidden_dim=128, seed=42):
        super(CriticNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.seed = seed
        
        self.network = nn.Sequential(
            nn.Linear(input_shape, hidden_dim),
            nn.ReLU(),
            nn.GRUCell(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, 1)
        )
        
        torch.manual_seed(self.seed)
        nn.init.normal_(self.network[0].weight, mean=0, std=0.1)
        nn.init.normal_(self.network[3].weight, mean=0, std=0.1)

        # 目标网络
        self.target_network = nn.Sequential(
            nn.Linear(input_shape, hidden_dim),
            nn.ReLU(),
            nn.GRUCell(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, inputs, hidden_state, use_target=False):
        # inputs = ensure_tensor(inputs)
        # hidden_state = ensure_tensor(hidden_state)
        assert type(use_target) == bool
        b, e = inputs.shape
        inputs = inputs.view(-1, e)
        if use_target == False:
            x = F.relu(self.network[0](inputs), inplace=True)            
            h_in = hidden_state.reshape(-1, self.hidden_dim)
            hh = self.network[2](x, h_in)
            values = self.network[3](hh)
        else:
            x = F.relu(self.target_network[0](inputs), inplace=True)
            h_in = hidden_state.reshape(-1, self.hidden_dim)
            hh = self.target_network[2](x, h_in)
            values = self.target_network[3](hh)
        return values.view(b, -1), hh.view(b, -1)

    def init_hidden(self, batch_size=1):
        return self.network[0].weight.new(batch_size, self.hidden_dim).zero_()

    def random_init_hidden(self, seed, batch_size=1):
        torch.manual_seed(seed)
        weights = torch.empty(batch_size, self.hidden_dim)
        nn.init.normal_(weights, mean=0, std=0.01)
        weights.requires_grad = True
        return weights

    def update_target_network(self):
        self.target_network.load_state_dict(self.network.state_dict(), strict=False)
    
    def cuda(self):
        self.network.cuda()
        self.target_network.cuda()

class PolicyRNN(nn.Module):
    """
    Recurrent neural network for computing conditional action probabilities.
    Args:
        obs_input_shape (tuple): Shape of the observation input.
        emb_input_shape (tuple): Shape of the embedding input.
        output_shape (int): Number of possible actions.
        num_agents (int): Number of agents.
        device (str, optional): Device to use ('cuda' or 'cpu'). Default is 'cuda'.
        optimizer (str, optional): Optimizer to use ('adam' or 'rmsprop'). Default is 'adam'.
        hidden_dim (int, optional): Dimension of the hidden layers. Default is 128.
        seed (int, optional): Random seed. Default is 42.
    Methods:
        init_hidden(batch_size=1):
            Initializes the hidden states for the policy and critic networks.
        random_init_hidden(seed, batch_size=1):
            Initializes the hidden states for the policy and critic networks with a random seed.
        forward(observation, state, policy_hidden_state, critic_hidden_state):
            Forward pass through the policy and critic networks.
        select_actions(ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
            Selects actions based on the current policy.
        predict_policy(observation, policy_hidden_state):
            Predicts the policy logits given an observation and hidden state.
        predict_value(state, critic_hidden_state):
            Predicts the value given a state and hidden state.
        train_network(batch, gamma=0.99, value_loss_weight=0.5, max_grad_norm=10.0):
            Trains the policy and critic networks using a batch of data.
        update_target_network():
            Updates the target network for the critic.
    """
    """Recurrent neural network for computing conditional action probabilities."""
    def __init__(self, obs_input_shape, emb_input_shape, output_shape, num_agents, device='cuda', optimizer='adam', hidden_dim=128, seed=42):
        super(PolicyRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_actions = output_shape
        self.seed = seed
        self.num_agents = num_agents
        self.num_actions = output_shape
        self.policy_network = PolicyNetwork(obs_input_shape, output_shape, hidden_dim, seed)
        self.critic_network = CriticNetwork(emb_input_shape, hidden_dim, seed)
        self.device = "cuda" if torch.cuda.is_available() and device=='cuda' else "cpu"
        if self.device == 'cuda':
            self.policy_network.cuda()
            self.critic_network.cuda()
        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(list(self.policy_network.parameters()) + list(self.critic_network.parameters()), lr=1e-3)
        elif optimizer == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(list(self.policy_network.parameters()) + list(self.critic_network.parameters()), lr=1e-3)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
    def get_hidden_states(self, bs_id=None):
        if bs_id is None:
            return self.policy_hidden, self.critic_hidden
        else:
            return self.policy_hidden[bs_id], self.critic_hidden[bs_id]
    def init_hidden(self, batch_size=1):
        self.policy_hidden = self.policy_network.init_hidden(batch_size, self.num_agents)
        self.critic_hidden = self.critic_network.init_hidden(batch_size)
        if self.device == 'cuda':
            self.policy_hidden = self.policy_hidden.cuda()
            self.critic_hidden = self.critic_hidden.cuda()
        return self.policy_hidden, self.critic_hidden

    def random_init_hidden(self, seed, batch_size=1):
        self.policy_weights = self.policy_network.random_init_hidden(seed, batch_size, self.num_agents)
        self.critic_weights = self.critic_network.random_init_hidden(seed, batch_size)
        if self.device == 'cuda':
            self.policy_hidden = self.policy_hidden.cuda()
            self.critic_hidden = self.critic_hidden.cuda()
        return self.policy_weights, self.critic_weights

    def forward(self, state, observation, policy_hidden_state, critic_hidden_state):
        """
        Perform a forward pass through the policy and critic networks. No reshapeing is needed for the input tensors.
        Args:
            observation (torch.Tensor): The input observation tensor of shape (batch_size, num_agents, observation_dim).
            state (torch.Tensor): The input state tensor of shape (batch_size, state_dim).
            policy_hidden_state (torch.Tensor): The hidden state tensor for the policy network of shape (batch_size, num_agents, hidden_dim).
            critic_hidden_state (torch.Tensor): The hidden state tensor for the critic network of shape (batch_size, hidden_dim).
        Returns:
            tuple: A tuple containing:
                - policy_logits (torch.Tensor): The output logits from the policy network of shape (batch_size, num_agents, output_dim).
                - predicted_values (torch.Tensor): The predicted values from the critic network of shape (batch_size, 1).
                - policy_hh (torch.Tensor): The updated hidden state from the policy network of shape (batch_size, hidden_dim).
                - critic_hh (torch.Tensor): The updated hidden state from the critic network of shape (batch_size, hidden_dim).
        """
        # b_o, a, e_o = observation.shape
        # b_s, e_s = state.shape
        # observation = observation.view(-1, e_o)
        # state = state.view(-1, e_s)
        observation = self.ensure_tensor(observation)
        state = self.ensure_tensor(state)
        policy_hidden_state = self.ensure_tensor(policy_hidden_state)
        critic_hidden_state = self.ensure_tensor(critic_hidden_state)
        policy_logits, policy_hh = self.policy_network(observation, policy_hidden_state.view(-1, self.hidden_dim))
        predicted_values, critic_hh = self.critic_network(state, critic_hidden_state.view(-1, self.hidden_dim))
        
        # return policy_logits.view(b_o, a, -1), predicted_values.view(b_s, a, -1), policy_hh.view(b_o, a, -1), critic_hh.view(b_s, a, -1)
        return policy_logits, predicted_values, policy_hh, critic_hh

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_skills"][:, t_ep]
        agent_inputs = ep_batch["obs"][:, t_ep]
        policy_hidden_state, _ = self.init_hidden()
        agent_outputs, _, _, _ = self.forward(agent_inputs, policy_hidden_state, None)
        chosen_actions = torch.argmax(agent_outputs, dim=-1)
        return chosen_actions

    def predict_policy(self, observation, policy_hidden_state):
        observation = self.ensure_tensor(observation)
        policy_hidden_state = self.ensure_tensor(policy_hidden_state)
        # batch_size, num_agents, _ = observation.shape
        # stacked_observations = observation.view(-1, observation.shape[-1])
        # stacked_policy_hidden_states = policy_hidden_state.view(-1, policy_hidden_state.shape[-1])
        with torch.no_grad():
            # policy_logits, new_policy_hidden_state = self.policy_network(stacked_observations, stacked_policy_hidden_states)
            policy_logits, new_policy_hidden_state = self.policy_network(observation, policy_hidden_state)
        # return policy_logits.view(batch_size, num_agents, -1), new_policy_hidden_state.view(batch_size, num_agents, -1)
        return policy_logits, new_policy_hidden_state

    def predict_value(self, state, critic_hidden_state, use_target=False):
        state = self.ensure_tensor(state)
        critic_hidden_state = self.ensure_tensor(critic_hidden_state)
        
        with torch.no_grad():
            value, new_critic_hidden_state = self.critic_network(state, critic_hidden_state, use_target)
        return value, new_critic_hidden_state

    def train_network(self, batch, gamma=0.99, value_loss_weight=0.5, max_grad_norm=10.0, use_real_data = False):
        """
        Trains the network using the provided batch of data.
        Args:
            batch (dict): A batch of data containing states, observations, actions, rewards, next_states, 
                          experienced_thresholds, improved_policy_probs, policy_hidden_states, critic_hidden_states, 
                          transformed_advantages, and sampled_actions.
            gamma (float, optional): Discount factor for future rewards. Default is 0.99.
            value_loss_weight (float, optional): Weight for the value loss in the total loss calculation. Default is 0.5.
            max_grad_norm (float, optional): Maximum norm for gradient clipping. Default is 10.0.
        Returns:
            float: The total loss value after the training step.
        """
        states, observations, actions, rewards, next_states, dones, experienced_thresholds, improved_policy_probs, policy_hidden_states, critic_hidden_states, transformed_advantages, sampled_actions = prepare_batch_data(batch, use_real_data=use_real_data)
        observations = torch.FloatTensor(observations).to(self.device)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        improved_policy_probs = torch.FloatTensor(improved_policy_probs).to(self.device)
        policy_hidden_states = torch.FloatTensor(policy_hidden_states).to(self.device)
        critic_hidden_states = torch.FloatTensor(critic_hidden_states).to(self.device)
        transformed_advantages = torch.FloatTensor(transformed_advantages).to(self.device)
        experienced_thresholds = torch.FloatTensor(experienced_thresholds).to(self.device)
        # [B, k, num_agents]
        sampled_actions = torch.LongTensor(sampled_actions).to(self.device)
        # shape of policy_logits is [B, num_agents, n_actions]
        policy_logits, predicted_values, _, new_critic_hidden_states = self.forward(states, observations, policy_hidden_states, critic_hidden_states)
        
        # Compute the probabilities for each action for each agent
        policy_probs = F.softmax(policy_logits, dim=-1)  # [B, num_agents, n_actions]
        
        ''' Explanation of gathered_probs computation:
            - policy_probs: [B, num_agents, n_actions]
            - sampled_actions: [B, k, num_agents] where k is the number of sampled actions per agent
            - policy_probs.unsqueeze(1): [B, 1, num_agents, n_actions]
            - sampled_actions.unsqueeze(-1): [B, k, num_agents, 1]
            - policy_probs.unsqueeze(1).expand(-1, sampled_actions.size(1), -1, -1): [B, k, num_agents, n_actions]
            - torch.gather(..., 3, sampled_actions.unsqueeze(-1)): [B, k, num_agents, 1]
            - .squeeze(-1): [B, k, num_agents]
            The purpose of this computation is to gather the probabilities corresponding to the sampled actions for each agent.
            This allows us to evaluate the policy's performance on the sampled actions by multiplying the probabilities of the sampled actions
            to get the combined probabilities for each sample.
        '''
        # 对于hier ma gumbel muzero来说，sampled_actions中的每一个，其实都是一个skill。skill的解码交给另外的解码器进行.做到最后，是可以通过policy直接获得skill而不需要mcts的
        gathered_probs = torch.gather(policy_probs.unsqueeze(1).expand(-1, sampled_actions.size(1), -1, -1), 3, sampled_actions.unsqueeze(-1)).squeeze(-1)
        gathered_logits = torch.gather(policy_logits.unsqueeze(1).expand(-1, sampled_actions.size(1), -1, -1), 3, sampled_actions.unsqueeze(-1)).squeeze(-1)
        # Compute the combined probabilities for each sampled action combination
        combo_probs = gathered_probs.prod(dim=2)  # [B, num_agents]
        combo_logits = gathered_logits.sum(dim=2)  # [B, num_agents]
        # Compute the policy loss. 这里是可以不使用去常数的logits的，因为experienced_thresholds是gumbel perturbed value，本身也包含常数，所以就消掉了
        experience_sample_probs = (1 - torch.exp(-torch.exp(combo_logits - experienced_thresholds))).detach()
        # print("experience_sample_probs.min = ", experience_sample_probs.min().detach().cpu().numpy())
        normalized_factor = 1 + torch.sum(combo_probs * (torch.exp(transformed_advantages) - 1), axis=-1, keepdim=True)
        policy_prob_mpo = ((combo_probs * torch.exp(transformed_advantages) / normalized_factor)).detach()
        policy_loss = -torch.mean(torch.sum(policy_prob_mpo / (experience_sample_probs+1e-8) * torch.log(combo_probs + 1e-8), axis=-1))
        
        with torch.no_grad():
            target_values, _ = self.predict_value(next_states, new_critic_hidden_states.view(-1, self.hidden_dim), use_target=True)
            target_values = rewards + gamma * target_values.squeeze(-1) * (1 - dones)
        
        value_loss = nn.MSELoss()(predicted_values.squeeze(-1), target_values)
        # 总损失 (可以调整权重)
        total_loss = policy_loss + value_loss_weight * value_loss
        
        # 更新策略网络参数
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.policy_network.parameters()) + list(self.critic_network.parameters()), max_grad_norm)
        self.optimizer.step()
        loss_dict = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "total_loss": total_loss.item()
        }
        return loss_dict

    def update_target_network(self):
        self.critic_network.update_target_network()

    def ensure_tensor(self, variable, dtype=torch.float32):
        if not isinstance(variable, torch.Tensor):
            variable = np.array(variable)
            variable = torch.tensor(variable, dtype=dtype, device=self.device)
        return variable


class ReplayBuffer:
    def __init__(self, capacity: int, batch_size: int, c_steps: int = 1, use_real_data = False):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.batch_size = batch_size
        self.use_real_data = use_real_data

    def push(self, policy_output, experienced_thresholds, advantages, root_policy_hidden_state, root_critic_hidden_state, real_r=None, real_next_obs=None, real_next_state=None, real_done = None):
        assert root_critic_hidden_state.shape[0] == self.batch_size
        if self.use_real_data == False:
            batch_data = (policy_output, experienced_thresholds, advantages, root_policy_hidden_state, root_critic_hidden_state)
        else:
            assert real_next_obs.shape[0] == self.batch_size
            assert real_next_state.shape[0] == self.batch_size
            batch_data = (policy_output, experienced_thresholds, advantages, root_policy_hidden_state, root_critic_hidden_state, real_r, real_next_obs, real_next_state, real_done)
        # 如果buffer未满，则直接添加数据
        # 如果buffer已满，则覆盖最旧的数据
        if len(self.buffer) < self.capacity:
            self.buffer.append(batch_data)
        else:
            self.buffer[self.position] = batch_data
        self.position = (self.position + 1) % self.capacity

    # def sample(self, batch_size: int) -> Tuple:
    #     indices = np.random.choice(len(self.buffer), batch_size, replace=False)
    #     batch = [self.buffer[i] for i in indices]
    #     return batch
    
    def clear(self):
        self.buffer = []
        self.position = 0

    def __len__(self):
        return len(self.buffer)


class ReplayBufferList:
    def __init__(self, capacity: int, use_real_data = False):
        self.capacity = capacity
        self.replay_buffer_list = []
        self.position = 0
        self.use_real_data = use_real_data

    def push(self, replay_buffer):
        # 支持传入单个 ReplayBuffer 或者包含多个 ReplayBuffer 的 list
        buffers = replay_buffer if isinstance(replay_buffer, list) else [replay_buffer]
        for buf in buffers:
            if len(self.replay_buffer_list) < self.capacity:
                self.replay_buffer_list.append(buf)
            else:
                self.replay_buffer_list[self.position] = buf
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> list:
        sample_size = max(batch_size // self.replay_buffer_list[0].batch_size, 1)
        assert len(self.replay_buffer_list) >= batch_size, "No enough data to sample."
        indices = np.random.choice(len(self.replay_buffer_list), sample_size, replace=False)
        sampled_data = []
        for i in indices:
            sampled_data.extend(self.replay_buffer_list[i].buffer)
        return sampled_data
    
    def clear(self):
        self.replay_buffer_list = []
        self.position = 0

    def __len__(self):
        return len(self.replay_buffer_list)

def compute_prior_from_qvalues(q_values: np.ndarray, temperature: float = 0.5, max_min_transform: bool = True) -> np.ndarray:
    """从Q值计算动作先验概率"""
    if max_min_transform:
        q_values = (q_values - np.min(q_values, axis=-1, keepdims=True)) / (np.max(q_values, axis=-1, keepdims=True) - np.min(q_values, axis=-1, keepdims=True))
    return np.exp(q_values / temperature) / np.sum(np.exp(q_values / temperature), axis=-1, keepdims=True)

def prepare_batch_data(sampled_batch: Tuple,
                       max_visit_init: float = 50.0,
                       value_scale: float = 0.1,
                       use_real_data: bool = False) -> Tuple:
    """
    Unify conversion of all torch.Tensor to numpy and collect batch data.
    Returns:
        states, observations, actions, rewards, next_states, dones,
        experienced_thresholds, improved_policy_probs,
        policy_hidden_states, critic_hidden_states,
        advantages, sampled_actions_list
    """

    def to_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.array(x)

    # prepare containers
    states = []
    observations = []
    actions = []
    rewards = []
    next_states = []
    dones = []
    experienced_thresholds = []
    improved_policy_probs = []
    policy_hidden_states = []
    critic_hidden_states = []
    advantages = []
    sampled_actions_list = []

    for entry in sampled_batch:
        if not use_real_data:

            policy_output, experienced_threshold, advantage, \
                root_policy_h, root_critic_h = entry
        else:
            (policy_output, experienced_threshold, advantage,
             root_policy_h, root_critic_h,
             real_r, real_next_obs, real_next_state, real_done) = entry


        tree = policy_output.search_tree
        action = policy_output.action
        action_weights = policy_output.action_weights

        root_idx = tree_lib.Tree.ROOT_INDEX
        batch_range = np.arange(tree.embeddings.shape[0])

        # sample data from the tree
        sampled_actions = to_np(tree.sampled_actions[:, root_idx])
        visit_count = tree.children_visits[batch_range, root_idx]
        max_visit = np.max(visit_count, axis=-1, keepdims=True)
        visit_scale = max_visit + max_visit_init
        transformed_adv = visit_scale * value_scale * advantage
        if not use_real_data:

            state_np = to_np(tree.embeddings[:, root_idx])
            obs_np = to_np(tree.observations[:, root_idx])
            reward_np = np.array([tree.children_rewards[b, root_idx, a]
                                  for b, a in zip(batch_range, action)])
            next_state_np = np.array([tree.embeddings[br, tree.children_index[br, root_idx, a]] for br, a in zip(batch_range, action)])
            done_np = np.zeros_like(reward_np, dtype=bool)

        else:

            # real data path
            state_np = to_np(real_next_state)
            obs_np = to_np(real_next_obs)
            reward_np = [to_np(real_r)]
            next_state_np = to_np(real_next_state)
            done_np = [to_np(real_done)]

        # append to lists
        states.append(state_np)
        observations.append(obs_np)
        actions.append(to_np(action))
        rewards.append(reward_np)
        next_states.append(next_state_np)
        dones.append(done_np)
        experienced_thresholds.append(to_np(experienced_threshold))
        improved_policy_probs.append(to_np(action_weights))
        policy_hidden_states.append(to_np(root_policy_h))
        critic_hidden_states.append(to_np(root_critic_h))
        advantages.append(to_np(transformed_adv))
        sampled_actions_list.append(sampled_actions)

    # concatenate along batch dimension
    states = np.concatenate(states, axis=0)
    observations = np.concatenate(observations, axis=0)
    actions = np.concatenate(actions, axis=0)
    rewards = np.concatenate(rewards, axis=0).flatten()
    next_states = np.concatenate(next_states, axis=0)
    dones = np.concatenate(dones, axis=0).flatten()
    experienced_thresholds = np.concatenate(experienced_thresholds, axis=0)
    improved_policy_probs = np.concatenate(improved_policy_probs, axis=0)
    policy_hidden_states = np.concatenate(policy_hidden_states, axis=0)
    critic_hidden_states = np.concatenate(critic_hidden_states, axis=0)
    advantages = np.concatenate(advantages, axis=0)
    sampled_actions_list = np.concatenate(sampled_actions_list, axis=0)

    return (states,
            observations,
            actions,
            rewards,
            next_states,
            dones,
            experienced_thresholds,
            improved_policy_probs,
            policy_hidden_states,
            critic_hidden_states,
            advantages,
            sampled_actions_list)
