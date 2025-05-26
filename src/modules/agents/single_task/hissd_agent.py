import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.transformer import Transformer
from src.modules.agents.multi_task.vq_skill import SkillModule, MLPNet

class RecModule(nn.Module):
    def __init__(self, args):
        super(RecModule, self).__init__()
        self.args = args
        self.entity_embed_dim = args.entity_embed_dim

        # 观察编码器: (obs_shape) -> (entity_embed_dim)
        self.obs_encoder = nn.Sequential(
            nn.Linear(args.obs_shape, 128),
            nn.ReLU(),
            nn.Linear(128, self.entity_embed_dim)
        )

        # 状态编码器: (state_shape) -> (entity_embed_dim)
        self.state_encoder = nn.Sequential(
            nn.Linear(args.state_shape, 128),
            nn.ReLU(),
            nn.Linear(128, self.entity_embed_dim)
        )

        # 输入投射层，合并 (obs_embed + skill_embed_per_agent)
        # obs_embed: entity_embed_dim, skill_embed_per_agent: entity_embed_dim
        self.input_projection = MLPNet(
            self.entity_embed_dim * 2, 
            self.entity_embed_dim, 
            hidden_dim=128, # Or args.hidden_dim if available
            num_layers=2
        )

        # Transformer 用于处理智能体序列
        self.transformer = Transformer(
            self.entity_embed_dim,
            args.head,
            args.depth,
            self.entity_embed_dim
        )

        # 下一观察预测头: (entity_embed_dim) -> (obs_shape)
        self.next_obs_pred_head = MLPNet(
            self.entity_embed_dim,
            args.obs_shape,
            hidden_dim=128, # Or args.hidden_dim
            num_layers=2,
            output_norm=False
        )

        # 下一状态预测头: (pooled_agent_embed + state_embed + global_skill_embed) -> (state_shape)
        # Each component is entity_embed_dim, so input is entity_embed_dim * 3
        self.next_state_pred_head = MLPNet(
            self.entity_embed_dim * 3,
            args.state_shape,
            hidden_dim=256, # Or args.hidden_dim * 2
            num_layers=2,
            output_norm=False
        )

    def pred_next(self, obs, state, global_skill_emb):
        """
        预测下一个观察和状态。
        obs: (bs * n_agents, obs_shape) - 当前观察
        state: (bs, state_shape) - 当前状态
        global_skill_emb: (bs, entity_embed_dim) - 全局技能嵌入
        """
        bs = state.shape[0]
        n_agents = self.args.n_agents

        # 1. 编码观察
        # (bs * n_agents, obs_shape) -> (bs * n_agents, entity_embed_dim)
        encoded_obs = self.obs_encoder(obs)

        # 2. 为每个智能体准备技能嵌入
        # (bs, entity_embed_dim) -> (bs, n_agents, entity_embed_dim) -> (bs * n_agents, entity_embed_dim)
        tiled_skill_emb = global_skill_emb.unsqueeze(1).repeat(1, n_agents, 1).reshape(bs * n_agents, self.entity_embed_dim)

        # 3. 合并编码后的观察和技能嵌入，然后投射
        # (bs * n_agents, entity_embed_dim * 2)
        combined_agent_input = th.cat([encoded_obs, tiled_skill_emb], dim=-1)
        # (bs * n_agents, entity_embed_dim)
        projected_agent_input = self.input_projection(combined_agent_input)

        # 4. Reshape for Transformer
        # (bs * n_agents, entity_embed_dim) -> (bs, n_agents, entity_embed_dim)
        transformer_input = projected_agent_input.reshape(bs, n_agents, self.entity_embed_dim)

        # 5. 通过 Transformer 处理
        # (bs, n_agents, entity_embed_dim)
        transformer_output = self.transformer(transformer_input, None)

        # 6. 预测下一观察
        # (bs, n_agents, entity_embed_dim) -> (bs * n_agents, entity_embed_dim)
        next_obs_transformer_flat = transformer_output.reshape(bs * n_agents, self.entity_embed_dim)
        # (bs * n_agents, obs_shape)
        pred_next_obs = self.next_obs_pred_head(next_obs_transformer_flat)

        # 7. 预测下一状态
        # (bs, state_shape) -> (bs, entity_embed_dim)
        encoded_state = self.state_encoder(state)
        # (bs, n_agents, entity_embed_dim) -> (bs, entity_embed_dim)
        pooled_agent_representation = transformer_output.mean(dim=1)
        
        # (bs, entity_embed_dim * 3)
        state_predictor_input = th.cat([pooled_agent_representation, encoded_state, global_skill_emb], dim=-1)
        # (bs, state_shape)
        pred_next_state = self.next_state_pred_head(state_predictor_input)
        
        return pred_next_obs, pred_next_state
        
    def forward(self, skill_emb, obs, next_obs, state, next_state, actions=None):
        """
        计算重构损失。
        skill_emb: (bs, entity_embed_dim) - 来自 PlannerModel 的全局技能输出
        obs: (bs * n_agents, obs_shape) - 当前观察
        next_obs: (bs * n_agents, obs_shape) - 真实的下一观察 (target)
        state: (bs, state_shape) - 当前状态
        next_state: (bs, state_shape) - 真实的下一状态 (target)
        """
        # 预测下一个观察和状态
        # obs 是当前时间步的观察，state 是当前时间步的状态
        pred_next_obs, pred_next_state = self.pred_next(obs, state, skill_emb)
        
        # 计算MSE损失
        # next_obs 是目标下一观察，next_state 是目标下一状态
        obs_loss = F.mse_loss(pred_next_obs, next_obs.detach())
        state_loss = F.mse_loss(pred_next_state, next_state.detach())
        
        # 总损失
        loss = obs_loss + state_loss
        return loss

class ValueNet(nn.Module):
    def __init__(self, input_shape, args):
        super(ValueNet, self).__init__()
        self.args = args
        self.entity_embed_dim = args.entity_embed_dim
        
        # 观察编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_shape, 256),
            nn.ReLU(),
            nn.Linear(256, self.entity_embed_dim)
        )
        
        # Transformer处理编码后的观察
        self.transformer = Transformer(
            self.entity_embed_dim, 
            args.head, 
            args.depth, 
            self.entity_embed_dim
        )
        
        # 价值头
        self.reward_fc = nn.Sequential(
            nn.Linear(self.entity_embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, inputs, hidden_state, actions=None):
        """前向传播计算价值"""
        # 编码输入
        x = self.encoder(inputs)
        
        # 合并编码输入和隐藏状态
        if hidden_state is not None:
            hidden = hidden_state.reshape(-1, 1, self.entity_embed_dim)
            total_hidden = th.cat([x.unsqueeze(1), hidden], dim=1)
        else:
            total_hidden = x.unsqueeze(1)
            
        # 通过Transformer处理
        outputs = self.transformer(total_hidden, None)
        h = outputs[:, -1]  # 新的隐藏状态
        reward = outputs[:, 0]  # 输出价值
        reward = self.reward_fc(reward)
        
        return reward, h
        
    def forward_skill(self, inputs, hidden_state):
        """使用技能表示计算价值"""
        if hidden_state is not None:
            hidden = hidden_state.reshape(-1, 1, self.entity_embed_dim)
            total_hidden = th.cat([inputs, hidden], dim=1)
        else:
            total_hidden = inputs
            
        outputs = self.transformer(total_hidden, None)
        h = outputs[:, -1]
        reward = outputs[:, 0]
        reward = self.reward_fc(reward)
        
        return reward, h

class Decoder(nn.Module):
    def __init__(self, input_shape, n_actions, args):
        super(Decoder, self).__init__()
        self.args = args
        self.entity_embed_dim = args.entity_embed_dim
        
        # 观察编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_shape, 256),
            nn.ReLU(),
            nn.Linear(256, self.entity_embed_dim)
        )
        
        # Transformer处理编码后的观察和技能
        self.transformer = Transformer(
            self.entity_embed_dim, 
            args.head, 
            args.depth, 
            self.entity_embed_dim
        )
        
        # 动作预测头
        self.action_head = MLPNet(
            self.entity_embed_dim,
            n_actions,
            256,
            2,
            output_norm=False
        )
        
    def forward(self, skill, inputs, hidden_state, actions=None):
        """前向传播解码为动作"""
        # 编码输入
        x = self.encoder(inputs)
        
        # 合并技能、编码输入和隐藏状态
        if hidden_state is not None:
            hidden = hidden_state.reshape(-1, 1, self.entity_embed_dim)
            if skill is not None:
                total_hidden = th.cat([skill.unsqueeze(1), x.unsqueeze(1), hidden], dim=1)
            else:
                total_hidden = th.cat([x.unsqueeze(1), hidden], dim=1)
        else:
            if skill is not None:
                total_hidden = th.cat([skill.unsqueeze(1), x.unsqueeze(1)], dim=1)
            else:
                total_hidden = x.unsqueeze(1)
                
        # 通过Transformer处理
        outputs = self.transformer(total_hidden, None)
        h = outputs[:, -1]  # 新的隐藏状态
        action_out = self.action_head(outputs[:, 0])  # 动作输出
        
        return action_out, h

class PlannerModel(nn.Module):
    def __init__(self, input_shape, args):
        super(PlannerModel, self).__init__()
        self.args = args
        self.entity_embed_dim = args.entity_embed_dim
        self.skill_dim = args.skill_dim
        self.vq_skill = args.vq_skill
        
        # 观察编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_shape, 256),
            nn.ReLU(),
            nn.Linear(256, self.entity_embed_dim)
        )
        
        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(args.state_shape, 256),
            nn.ReLU(),
            nn.Linear(256, self.entity_embed_dim)
        )
        
        # Transformer处理编码后的观察
        self.transformer = Transformer(
            self.entity_embed_dim, 
            args.head, 
            args.depth, 
            self.entity_embed_dim
        )
        
        # 技能模块处理VQ-VAE
        self.skill_module = SkillModule(args)
        
        # 重构模块预测下一个观察和状态
        self.rec_module = RecModule(args)
        
        # 技能和不同目的的整合网络
        self.act_forward = MLPNet(2*self.entity_embed_dim, self.entity_embed_dim, 128)
        self.value_forward = MLPNet(2*self.entity_embed_dim, self.entity_embed_dim, 128)
        self.rew_forward = MLPNet(2*self.entity_embed_dim, self.entity_embed_dim, 128)
    
    def forward(self, inputs, states, hidden_state=None, actions=None, 
                next_inputs=None, next_states=None, loss_out=False, 
                skill_index_out=False, training=True, external_skill_index=None):
        """前向传播计算技能表示"""
        bs_times_n_agents = inputs.shape[0]
        bs = states.shape[0]
        n_agents = self.args.n_agents if bs > 0 else 1 # Avoid division by zero if bs is 0
        if bs_times_n_agents > 0 and bs > 0 :
            assert bs_times_n_agents // bs == n_agents, "Batch size mismatch"


        # 编码输入
        x = self.encoder(inputs) # (bs*n_agents, entity_embed_dim)
        # state_emb = self.state_encoder(states) # (bs, entity_embed_dim) # Not directly used for skill generation here
        
        # 合并编码输入和隐藏状态
        if hidden_state is not None:
            # hidden_state is (bs*n_agents, entity_embed_dim)
            hidden = hidden_state.reshape(bs_times_n_agents, 1, self.entity_embed_dim)
            total_hidden = th.cat([x.unsqueeze(1), hidden], dim=1) # (bs*n_agents, 2, entity_embed_dim)
        else:
            total_hidden = x.unsqueeze(1) # (bs*n_agents, 1, entity_embed_dim)
            
        # 通过Transformer处理
        outputs = self.transformer(total_hidden, None) # (bs*n_agents, num_tokens, entity_embed_dim)
        h = outputs[:, -1]  # 新的隐藏状态 (bs*n_agents, entity_embed_dim)
        
        # 从每个智能体的Transformer输出中提取用于技能生成的输入
        agent_skill_features = outputs[:, 0]  # (bs*n_agents, entity_embed_dim)
        
        # 聚合智能体特征以形成全局技能输入
        if bs > 0:
            global_skill_input = agent_skill_features.reshape(bs, n_agents, self.entity_embed_dim).mean(dim=1) # (bs, entity_embed_dim)
        elif bs_times_n_agents > 0 : # Case where bs might be 1 implicitly if n_agents is bs_times_n_agents
             global_skill_input = agent_skill_features.mean(dim=0, keepdim=True) # (1, entity_embed_dim)
        else: # No inputs, create zero tensor
            global_skill_input = th.zeros(0, self.entity_embed_dim, device=inputs.device, dtype=inputs.dtype)


        # 通过技能模块处理
        commit_loss = th.tensor(0.).to(inputs.device)
        diver_loss = th.tensor(0.).to(inputs.device)
        vq_loss = th.tensor(0.).to(inputs.device) # For VQ loss if applicable
        skill_index = None
        
        skill_output_global = global_skill_input # Default if not VQ
        if global_skill_input.shape[0] > 0: # Proceed only if there's data
            if self.vq_skill:
                if external_skill_index is not None:
                    # TODO: 这个函数需要注意，这个是动作直接对应到离散的embedding，也就是gumbel muzero中，通过skill解码得到其他东西所需要的embedding。
                    # 当有real_world_simulator时，这个并不需要
                    skill_output_global, commit_loss, diver_loss = self.skill_module.forward_with_skill_index(
                        global_skill_input, external_skill_index
                    )
                    skill_index = external_skill_index
                else:
                    skill_output_global, skill_index, commit_loss, diver_loss, vq_loss = self.skill_module(
                        global_skill_input, training=training
                    )
            else:
                # 非VQ模式，直接使用转换后的表示 (already assigned)
                pass
        else: # Handle empty batch case for skill_output_global
            skill_output_global = th.zeros(0, self.entity_embed_dim, device=inputs.device, dtype=inputs.dtype)

        # 计算重构损失
        rec_loss = th.tensor(0.).to(inputs.device)
        if next_inputs is not None and loss_out and skill_output_global.shape[0] > 0:
            rec_loss = self.rec_module(
                skill_output_global, # Pass global skill (bs, entity_embed_dim)
                inputs, 
                next_inputs, 
                states, 
                next_states, 
                actions=actions
            )
            
        # 总损失
        out_loss = commit_loss + diver_loss + rec_loss + vq_loss
        
        # 损失字典
        loss_dict = {
            "commit_loss": commit_loss.detach().cpu().item(),
            "diver_loss": diver_loss.detach().cpu().item(),
            "rec_loss": rec_loss.detach().cpu().item(),
            "vq_loss": vq_loss.detach().cpu().item(),
        }
        
        # skill_output_global is (bs, entity_embed_dim)
        # h is (bs*n_agents, entity_embed_dim)
        return skill_output_global, h, out_loss, skill_index, loss_dict
    
    def feedforward(self, skill_emb, forward_type='action', additional_input=None, adaptation=False):
        """
        使用技能表示进行不同类型的前向传播。
        skill_emb: (bs, entity_embed_dim) - 全局技能嵌入
        additional_input: (bs * n_agents, obs_shape) or (bs, state_shape)
        """
        assert forward_type in ['action', 'value', 'reward']
        
        if additional_input is None:
            raise ValueError("additional_input should not be None")

        bs = skill_emb.shape[0]
        n_agents = self.args.n_agents
        
        # 根据前向类型处理输入
        if forward_type == 'reward' or forward_type == 'value':
            # additional_input is state (bs, state_shape)
            with th.no_grad() if adaptation else th.enable_grad():
                emb = self.state_encoder(additional_input) # (bs, entity_embed_dim)
            # Tile emb to match skill_emb if skill_emb is per agent, or tile skill_emb if emb is global
            # Here, skill_emb is (bs, D), emb is (bs, D). We need per-agent output.
            # So, tile both to (bs*n_agents, D) for consistency if MLP expects per-agent inputs.
            # However, the MLP (e.g. self.value_forward) takes 2*entity_embed_dim.
            # This implies concatenation of skill and emb.
            # If output is per agent, then skill_emb needs to be tiled.
            # If output is global (e.g. global value from global state), no tiling needed for skill_emb.
            # Let's assume the output of value/reward feedforward is global (per batch item).
            # So, skill_emb (bs,D) and emb (bs,D) are fine.
            # The MLPNet will output (bs, entity_embed_dim)
            tiled_skill_emb_for_cat = skill_emb
            processed_add_input_emb = emb

        elif forward_type == 'action':
            # additional_input is obs (bs*n_agents, obs_shape)
            with th.no_grad() if adaptation else th.enable_grad():
                emb = self.encoder(additional_input) # (bs*n_agents, entity_embed_dim)
            # Tile skill_emb (bs, D) to (bs*n_agents, D)
            if bs > 0:
                tiled_skill_emb_for_cat = skill_emb.unsqueeze(1).repeat(1, n_agents, 1).reshape(bs * n_agents, self.entity_embed_dim)
            elif additional_input.shape[0] > 0: # bs is 0, but additional_input might not be (e.g. single agent inference)
                 tiled_skill_emb_for_cat = skill_emb.repeat(additional_input.shape[0], 1)
            else: # No data
                tiled_skill_emb_for_cat = th.zeros(0, self.entity_embed_dim, device=skill_emb.device, dtype=skill_emb.dtype)

            processed_add_input_emb = emb
                
        # 根据前向类型整合技能和输入
        # Ensure inputs to cat have at least one data point if not empty
        if tiled_skill_emb_for_cat.shape[0] > 0 or processed_add_input_emb.shape[0] > 0:
            combined_input = th.cat([tiled_skill_emb_for_cat, processed_add_input_emb], dim=-1)
        else: # Handle case where both might be empty due to bs=0
            target_dim = tiled_skill_emb_for_cat.shape[-1] + processed_add_input_emb.shape[-1]
            combined_input = th.zeros(0, target_dim, device=skill_emb.device, dtype=skill_emb.dtype)


        if forward_type == 'action':
            out = self.act_forward(combined_input) # Output: (bs*n_agents, entity_embed_dim)
        elif forward_type == 'value':
            out = self.value_forward(combined_input) # Output: (bs, entity_embed_dim)
        elif forward_type == 'reward':
            out = self.rew_forward(combined_input) # Output: (bs, entity_embed_dim)
            
        return [out]

class HISSDAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(HISSDAgent, self).__init__()
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        
        # 维度
        self.skill_dim = args.skill_dim
        self.entity_embed_dim = args.entity_embed_dim
        
        # 核心组件
        self.value = ValueNet(input_shape, args)
        self.decoder = Decoder(input_shape, self.n_actions, args)
        self.planner = PlannerModel(input_shape, args)
        
        # 奖励预测
        self.reward_transformer = Transformer(
            args.entity_embed_dim, 
            args.head, 
            args.depth, 
            args.entity_embed_dim
        )
        self.reward_predict_net = nn.Sequential(
            nn.Linear(args.entity_embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # 存储上次规划输出
        self.last_out_h = None
        self.last_h_plan = None

    def init_hidden(self):
        """初始化隐藏状态"""
        device = next(self.parameters()).device
        return (
            th.zeros(1, self.args.entity_embed_dim, device=device),  # value
            th.zeros(1, self.args.entity_embed_dim, device=device),  # reward
            th.zeros(1, self.args.entity_embed_dim, device=device),  # decoder
            th.zeros(1, self.args.entity_embed_dim, device=device)   # planner
        )

    def forward_value(self, inputs, hidden_state_value, actions=None):
        """前向传播计算价值"""
        return self.value(inputs, hidden_state_value, actions)

    def forward_value_skill(self, inputs, hidden_state_value):
        """使用技能表示计算价值"""
        return self.value.forward_skill(inputs, hidden_state_value)

    def forward_reward_skill(self, inputs, hidden_state_reward):
        """使用技能表示计算奖励"""
        # 合并技能表示和隐藏状态
        total_hidden = th.cat(
            [inputs, hidden_state_reward.reshape(-1, 1, self.args.entity_embed_dim)], 
            dim=1
        )
        # 通过Transformer处理
        outputs = self.reward_transformer(total_hidden, None)
        h = outputs[:, -1]  # 新的隐藏状态
        reward = self.reward_predict_net(outputs[:, 0])  # 奖励输出
        
        return reward, h

    def forward_planner(self, inputs, states, hidden_state_plan=None,
                       actions=None, next_inputs=None, next_states=None, 
                       loss_out=False, skill_index_out=False, 
                       training=True, external_skill_index=None):
        """前向传播计算技能表示"""
        return self.planner(
            inputs, states, hidden_state_plan, actions,
            next_inputs, next_states, loss_out,
            skill_index_out, training, external_skill_index
        )

    def forward_planner_feedforward(self, emb_inputs, forward_type='action', 
                                  additional_input=None, adaptation=False):
        """使用技能表示进行不同类型的前向传播"""
        return self.planner.feedforward(
            emb_inputs, forward_type, additional_input, adaptation
        )

    def forward_contrastive(self, inputs, inputs_pos):
        """计算对比损失"""
        # 标准化向量以进行余弦相似度计算
        inputs_norm = F.normalize(inputs, dim=-1)
        inputs_pos_norm = F.normalize(inputs_pos, dim=-1)
        
        # 计算相似度矩阵
        logits = th.matmul(inputs_norm, inputs_pos_norm.transpose(0, 1))
        return logits

    def forward(self, inputs, states, hidden_state_dec, t, skill, 
               hidden_state_plan=None, actions=None, 
               skill_index_out=False, test_mode=None):
        """前向传播主函数"""
        # inputs: (bs*n_agents, obs_shape)
        # states: (bs, state_shape)
        # hidden_state_plan: (bs*n_agents, entity_embed_dim)
        # skill: Optional, (bs*n_agents, entity_embed_dim) - processed skill for decoder

        if t % self.args.c_step == 0:
            if skill is None: # skill here refers to the processed skill for decoder
                # 生成新的全局技能
                # planner_skill_raw: (bs, entity_embed_dim)
                # h_plan: (bs*n_agents, entity_embed_dim)
                planner_skill_raw, h_plan, _, skill_idx_from_planner, _ = self.forward_planner(
                    inputs, states, hidden_state_plan=hidden_state_plan, 
                    actions=actions, skill_index_out=skill_index_out # skill_index_out for planner's internal VQ index
                )
                
                # 处理全局技能以用于动作选择 (contextualize with current obs)
                # planner_skill_raw (bs, D) and inputs (bs*n_agents, obs)
                # contextualized_skill (bs*n_agents, D)
                contextualized_skill = self.forward_planner_feedforward(
                    planner_skill_raw, forward_type='action', additional_input=inputs
                )[0]
                
                # 存储以供之后使用
                self.last_out_h = contextualized_skill # This is the skill for the decoder
                self.last_h_plan = h_plan
                
                # Propagate skill_index if requested
                if skill_index_out:
                    current_skill_index = skill_idx_from_planner
                else:
                    current_skill_index = None
            else:
                # If skill is provided, it's already the contextualized one for the decoder
                self.last_out_h = skill 
                # h_plan would be stale if skill is provided, ideally it should be passed too or recomputed
                # For simplicity, if skill is given, we might not have a fresh h_plan unless it's also passed.
                current_skill_index = None # Or needs to be passed if skill is provided externally with its index
        else: # Not a planning step
            # current_skill_index is None because no new skill was planned
            current_skill_index = None
            # self.last_out_h and self.last_h_plan are reused from previous planning step

        # 解码技能为动作
        # self.last_out_h is (bs*n_agents, D)
        act, h_dec = self.decoder(
            self.last_out_h, inputs, hidden_state_dec, actions
        )
        
        if skill_index_out == False or t % self.args.c_step != 0 : # only return skill_index if it's a planning step and requested
            return act, self.last_h_plan, h_dec, None
        else:
            # current_skill_index is set during planning step if skill_index_out is True
            return act, self.last_h_plan, h_dec, current_skill_index

    def get_codebook(self):
        """返回planner的skill模块的codebook的权重"""
        if hasattr(self.planner, 'skill_module') and hasattr(self.planner.skill_module, 'emb'):
            return self.planner.skill_module.emb.weight # nn.Embedding.weight is (num_embeddings, embedding_dim)
        return None
        
    def get_skill(self, skill_index):
        """根据索引返回对应的解码后的技能表示"""
        if hasattr(self.planner, 'skill_module') and \
           hasattr(self.planner.skill_module, 'emb') and \
           hasattr(self.planner.skill_module, 'skill_decoder'):
            
            device = self.planner.skill_module.emb.weight.device
            # Ensure skill_index is a tensor on the correct device
            if not isinstance(skill_index, th.Tensor):
                skill_index = th.tensor(skill_index, device=device, dtype=th.int64)
            else:
                skill_index = skill_index.to(device=device, dtype=th.int64)

            original_shape = skill_index.shape
            skill_index_flat = skill_index.flatten()

            if skill_index_flat.numel() == 0: # Handle empty skill_index
                 return th.zeros(*original_shape, self.args.entity_embed_dim, device=device, dtype=self.planner.skill_module.emb.weight.dtype)

            # 获取codebook向量: (N, code_dim) where N is numel of skill_index_flat
            codebook_vectors = self.planner.skill_module.emb(skill_index_flat)
            
            # 解码为技能表示: skill_decoder expects (batch_size, code_dim)
            # Output: (N, entity_embed_dim)
            decoded_skill_flat = self.planner.skill_module.skill_decoder(codebook_vectors)
            
            # Reshape to original_shape + entity_embed_dim
            decoded_skill = decoded_skill_flat.reshape(*original_shape, self.args.entity_embed_dim)
            
            return decoded_skill
        return None
