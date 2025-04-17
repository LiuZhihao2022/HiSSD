import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

class MultiAgentWorldModel(nn.Module):
    def __init__(self, 
                 num_agents, 
                 obs_dim, 
                 action_dim, 
                 hidden_dim=128,
                 use_attention=True):
        super().__init__()
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        
        # 共享的编码器网络
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # 注意力机制处理智能体间交互
        self.attention = nn.MultiheadAttention(hidden_dim, 4) if use_attention else None
        
        # 动态预测网络 - 修改为预测所有智能体的观察
        self.transition = nn.Sequential(
            nn.Linear(hidden_dim * num_agents, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, obs_dim * num_agents)  # 输出所有智能体的观察
        )
        
        # 可选的奖励预测模块 - 修改为预测所有智能体的奖励
        self.reward_predictor = nn.Sequential(
            nn.Linear(hidden_dim * num_agents, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_agents)  # 每个智能体一个奖励值
        )

    def forward(self, obs, actions):
        # 输入形状处理
        batch_size = obs.shape[0]
        x = torch.cat([obs, actions], dim=-1)
        
        # 编码器处理
        encoded = self.encoder(x.view(-1, x.shape[-1]))
        encoded = encoded.view(batch_size, self.num_agents, -1)
        
        # 注意力机制
        if self.attention is not None:
            attn_out, _ = self.attention(encoded, encoded, encoded)
            context = attn_out.view(batch_size, -1)
        else:
            context = encoded.view(batch_size, -1)
        
        # 状态转移预测 - 改为多智能体输出
        next_obs_pred = self.transition(context)
        next_obs_pred = next_obs_pred.view(batch_size, self.num_agents, self.obs_dim)
        
        # 奖励预测（可选）- 改为多智能体输出
        reward_pred = self.reward_predictor(context)
        reward_pred = reward_pred.view(batch_size, self.num_agents)
        
        return next_obs_pred, reward_pred

class WorldModelTrainer:
    def __init__(self, config):
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        
        # 初始化模型
        self.model = MultiAgentWorldModel(
            num_agents=config["num_agents"],
            obs_dim=config["obs_dim"],
            action_dim=config["action_dim"],
            hidden_dim=config.get("hidden_dim", 128),
            use_attention=config.get("use_attention", True)
        ).to(self.device)
        
        # 优化器设置
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.get("lr", 1e-3))
        self.loss_fn = nn.MSELoss()
        
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        
        for batch in dataloader:
            obs, actions, next_obs, rewards = batch
            obs = obs.to(self.device)
            actions = actions.to(self.device)
            next_obs = next_obs.to(self.device)
            rewards = rewards.to(self.device).unsqueeze(-1) if rewards.dim() == 1 else rewards.to(self.device)
            
            # 前向传播
            pred_next, pred_rewards = self.model(obs, actions)
            
            # 计算损失
            transition_loss = self.loss_fn(pred_next, next_obs)
            
            # 确保奖励预测维度匹配
            if pred_rewards is not None:
                if rewards.dim() == 1:
                    rewards = rewards.unsqueeze(-1)
                if rewards.shape != pred_rewards.shape:
                    rewards = rewards.view(pred_rewards.shape)
                reward_loss = self.loss_fn(pred_rewards, rewards)
            else:
                reward_loss = 0
                
            total_batch_loss = transition_loss + reward_loss
            
            # 反向传播
            self.optimizer.zero_grad()
            total_batch_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # 梯度裁剪
            self.optimizer.step()
            
            total_loss += total_batch_loss.item()
            
        return total_loss / len(dataloader)

    def prepare_data(self, dataset):
        # 数据预处理
        # 假设dataset是字典形式，包含以下键：
        # observations, actions, next_observations, rewards
        obs_tensor = torch.FloatTensor(dataset["observations"])
        actions_tensor = torch.FloatTensor(dataset["actions"])
        next_obs_tensor = torch.FloatTensor(dataset["next_observations"])
        rewards_tensor = torch.FloatTensor(dataset["rewards"])
        
        # 创建数据集和数据加载器
        full_dataset = TensorDataset(obs_tensor, actions_tensor, next_obs_tensor, rewards_tensor)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        
        return DataLoader(train_dataset, batch_size=128, shuffle=True), \
               DataLoader(val_dataset, batch_size=128)

def load_mpe_dataset(env_name, data_split='expert', seed=0):
    """
    加载MPE数据集并转换为适合世界模型训练的格式
    
    Args:
        env_name: MPE环境名称
        data_split: 数据集划分 (默认为'expert')
        seed: 随机种子
        
    Returns:
        包含训练数据的字典
    """
    # 构建数据路径
    data_path = os.path.join(f"datasets/mpe/{env_name}/{data_split}/seed_{seed}_data")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset directory not found: {data_path}")
    
    print(f"Loading data from {data_path}")
    
    # 直接加载数据文件
    observations = np.load(os.path.join(data_path, "obs_0.npy"))
    actions = np.load(os.path.join(data_path, "acs_0.npy"))
    next_observations = np.load(os.path.join(data_path, "next_obs_0.npy"))
    rewards = np.load(os.path.join(data_path, "rews_0.npy"))
    dones = np.load(os.path.join(data_path, "dones_0.npy"))
    
    # 检查是否有多个智能体的数据文件
    agent_idx = 1
    all_observations = [observations]
    all_actions = [actions]
    all_next_observations = [next_observations]
    all_rewards = [rewards]
    
    # 尝试加载更多智能体的数据
    while os.path.exists(os.path.join(data_path, f"obs_{agent_idx}.npy")):
        all_observations.append(np.load(os.path.join(data_path, f"obs_{agent_idx}.npy")))
        all_actions.append(np.load(os.path.join(data_path, f"acs_{agent_idx}.npy")))
        all_next_observations.append(np.load(os.path.join(data_path, f"next_obs_{agent_idx}.npy")))
        all_rewards.append(np.load(os.path.join(data_path, f"rews_{agent_idx}.npy")))
        agent_idx += 1
    
    num_agents = agent_idx
    print(f"Found data for {num_agents} agents")
    
    # 如果有多个智能体，将数据重新排列为正确的形状
    if num_agents > 1:
        observations = np.stack(all_observations, axis=1)  # [T, N, obs_dim]
        actions = np.stack(all_actions, axis=1)            # [T, N, action_dim]
        next_observations = np.stack(all_next_observations, axis=1)  # [T, N, obs_dim]
        rewards = np.stack(all_rewards, axis=1)            # [T, N]
    
    # 收集所有样本（过滤掉 done=True 的状态）- 修复类型错误
    try:
        # 尝试安全地转换dones类型
        if isinstance(dones, np.ndarray):
            valid_indices = ~dones.astype(bool)
            if len(dones.shape) > 1:
                valid_indices = ~np.any(dones.astype(bool), axis=1)
        else:
            print(f"Warning: dones has unexpected type: {type(dones)}")
            valid_indices = np.ones(len(observations), dtype=bool)
    except Exception as e:
        print(f"Error processing dones: {e}")
        # 如果处理失败，保留所有数据
        valid_indices = np.ones(len(observations), dtype=bool)
    
    dataset = {
        'observations': observations[valid_indices],
        'actions': actions[valid_indices],
        'next_observations': next_observations[valid_indices],
        'rewards': rewards[valid_indices]
    }
    
    print(f"Loaded dataset with {len(dataset['observations'])} transitions")
    return dataset

def evaluate(model, dataloader):
    """
    评估世界模型性能
    """
    model.eval()
    total_loss = 0.0
    device = next(model.parameters()).device
    loss_fn = nn.MSELoss()
    
    with torch.no_grad():
        for batch in dataloader:
            obs, actions, next_obs, rewards = [b.to(device) for b in batch]
            if rewards.dim() == 1:
                rewards = rewards.unsqueeze(-1)
                
            pred_next, pred_rewards = model(obs, actions)
            
            transition_loss = loss_fn(pred_next, next_obs)
            
            if pred_rewards is not None:
                if rewards.shape != pred_rewards.shape:
                    rewards = rewards.view(pred_rewards.shape)
                reward_loss = loss_fn(pred_rewards, rewards)
            else:
                reward_loss = 0
                
            batch_loss = transition_loss + reward_loss
            
            total_loss += batch_loss.item()
    
    return total_loss / len(dataloader)

# 使用示例
if __name__ == "__main__":
    # 环境和数据集配置
    env_name = "simple_spread"  # 可选: simple_tag, simple_spread, simple_adversary等
    data_split = "expert"
    seed = 0
    
    # 加载MPE离线数据集
    try:
        print("尝试加载预处理的数据集...")
        dataset = np.load("mpe_dataset.npy", allow_pickle=True).item()
    except (FileNotFoundError, ValueError):
        print("预处理数据集不存在，从原始文件加载...")
        dataset = load_mpe_dataset(env_name, data_split=data_split, seed=seed)
        # 保存预处理数据以便将来使用
        np.save(f"datasets/mpe/{env_name}/{data_split}/seed_{seed}_dataset.npy", dataset)
    
    # 自动检测观察和动作空间维度
    obs_sample = dataset['observations'][0]  # 第一个样本
    act_sample = dataset['actions'][0]  # 第一个样本
    
    # 打印数据形状信息
    print(f"数据集样本数: {len(dataset['observations'])}")
    print(f"观察形状: {dataset['observations'].shape}")
    print(f"动作形状: {dataset['actions'].shape}")
    
    if len(obs_sample.shape) >= 2:
        num_agents = obs_sample.shape[0]
        obs_dim = obs_sample.shape[1]
    else:
        num_agents = 1
        obs_dim = obs_sample.shape[0]
    
    if len(act_sample.shape) >= 2:
        action_dim = act_sample.shape[1]
    else:
        action_dim = act_sample.shape[0]
    
    print(f"自动检测 - 智能体数量: {num_agents}")
    print(f"自动检测 - 观察维度: {obs_dim}")
    print(f"自动检测 - 动作维度: {action_dim}")
    
    config = {
        "num_agents": num_agents,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "hidden_dim": 256,
        "lr": 3e-4,
        "epochs": 100
    }
    
    # 初始化训练器
    trainer = WorldModelTrainer(config)
    
    # 准备数据
    train_loader, val_loader = trainer.prepare_data(dataset)
    
    # 训练循环
    for epoch in range(config["epochs"]):
        train_loss = trainer.train_epoch(train_loader)
        val_loss = evaluate(trainer.model, val_loader)
        print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f}")
