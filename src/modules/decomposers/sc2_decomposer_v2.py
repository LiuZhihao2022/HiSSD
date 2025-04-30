import torch as th
import numpy as np

class SC2DecomposerV2:
    def __init__(self, env):
        """
        env: smacv2.env.starcraft2.starcraft2_new.StarCraft2Env 实例
        """
        self.n_agents = env.n_agents
        self.n_enemies = env.n_enemies
        self.n_actions_move = env.n_actions_move
        self.n_fov_actions = env.n_fov_actions
        self.n_actions_no_attack = env.n_actions_no_attack
        self.n_actions = env.n_actions
        # 观测各部分维度
        self.move_feats_dim = env.get_obs_move_feats_size()
        self.enemy_feats_shape = env.get_obs_enemy_feats_size()  # (n_enemies, n_feats)
        self.ally_feats_shape = env.get_obs_ally_feats_size()    # (n_allies, n_feats)
        self.own_feats_dim = env.get_obs_own_feats_size()
        # 状态各部分维度
        self.ally_state_dim = env.get_ally_num_attributes()
        self.enemy_state_dim = env.get_enemy_num_attributes()
        self.state_size = env.get_state_size()
        self.obs_size = env.get_obs_size()
        # 兼容sc2_decomposer.py的属性
        self.own_obs_dim = self.move_feats_dim + self.own_feats_dim
        self.obs_nf_en = self.enemy_feats_shape[1]
        self.obs_nf_al = self.ally_feats_shape[1]
        self.move_feats = self.move_feats_dim
        self.enemy_feats = self.enemy_feats_shape[0] * self.enemy_feats_shape[1]
        self.ally_feats = self.ally_feats_shape[0] * self.ally_feats_shape[1]
        self.own_feats = self.own_feats_dim
        self.obs_dim = self.move_feats + self.enemy_feats + self.ally_feats + self.own_feats
        self.state_nf_en = self.enemy_state_dim
        self.state_nf_al = self.ally_state_dim
        self.enemy_state_dim_total = self.n_enemies * self.enemy_state_dim
        self.ally_state_dim_total = self.n_agents * self.ally_state_dim
        self.last_action_state_dim = self.n_agents * self.n_actions if hasattr(env, 'state_last_action') and env.state_last_action else 0
        self.state_last_action = True if hasattr(env, 'state_last_action') and env.state_last_action else False
        self.timestep_number_state_dim = 1 if hasattr(env, 'state_timestep_number') and env.state_timestep_number else 0
        self.state_timestep_number = True if hasattr(env, 'state_timestep_number') and env.state_timestep_number else False
        self.state_dim = self.enemy_state_dim_total + self.ally_state_dim_total + self.last_action_state_dim + self.timestep_number_state_dim

    def decompose_obs(self, obs_input):
        # obs_input: [batch, obs_size]
        move_feats = obs_input[:, :self.move_feats_dim]
        base = self.move_feats_dim
        enemy_feats = [obs_input[:, base + i * self.obs_nf_en:base + (i + 1) * self.obs_nf_en] for i in range(self.n_enemies)]
        base += self.obs_nf_en * self.n_enemies
        ally_feats = [obs_input[:, base + i * self.obs_nf_al:base + (i + 1) * self.obs_nf_al] for i in range(self.n_agents - 1)]
        base += self.obs_nf_al * (self.n_agents - 1)
        own_feats = obs_input[:, base:base + self.own_feats_dim]
        own_obs = th.cat([move_feats, own_feats], dim=-1)
        return own_obs, enemy_feats, ally_feats

    def decompose_state(self, state_input):
        # 兼容sc2_decomposer.py的返回格式
        # state_input: [batch, seq, state_dim]
        ally_states = [state_input[:, :, i * self.state_nf_al:(i + 1) * self.state_nf_al] for i in range(self.n_agents)]
        base = self.n_agents * self.state_nf_al
        enemy_states = [state_input[:, :, base + i * self.state_nf_en:base + (i + 1) * self.state_nf_en] for i in range(self.n_enemies)]
        base += self.n_enemies * self.state_nf_en
        last_action_states = [state_input[:, :, base + i * self.n_actions:base + (i + 1) * self.n_actions] for i in range(self.n_agents)] if self.last_action_state_dim > 0 else []
        base += self.n_agents * self.n_actions if self.last_action_state_dim > 0 else 0
        timestep_number_state = state_input[:, :, base:base+self.timestep_number_state_dim] if self.timestep_number_state_dim > 0 else []
        return ally_states, enemy_states, last_action_states, timestep_number_state

    def decompose_action_info(self, action_info):
        shape = action_info.shape
        if len(shape) > 2:
            action_info = action_info.reshape(np.prod(shape[:-1]), shape[-1])
        no_attack_action_info = action_info[:, :self.n_actions_no_attack]
        attack_action_info = action_info[:, self.n_actions_no_attack:self.n_actions_no_attack + self.n_enemies]
        no_attack_action_info = no_attack_action_info.reshape(*shape[:-1], self.n_actions_no_attack)
        attack_action_info = attack_action_info.reshape(*shape[:-1], self.n_enemies)
        bin_attack_info = th.sum(attack_action_info, dim=-1).unsqueeze(-1)
        compact_action_info = th.cat([no_attack_action_info, bin_attack_info], dim=-1)
        return no_attack_action_info, attack_action_info, compact_action_info
