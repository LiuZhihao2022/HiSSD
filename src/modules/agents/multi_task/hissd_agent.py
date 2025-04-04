import collections
import numpy as np
import torch as th
from torch.cuda import device_of
import torch.nn as nn
import torch.nn.functional as F
import h5py

from utils.embed import polynomial_embed, binary_embed
from utils.transformer import Transformer
from .vq_skill import SkillModule, MLPNet


class HISSDAgent(nn.Module):
    """  sotax agent for multi-task learning """

    def __init__(self, task2input_shape_info, task2decomposer, task2n_agents, decomposer, args):
        super(HISSDAgent, self).__init__()
        self.task2last_action_shape = {task: task2input_shape_info[task]["last_action_shape"] for task in
            task2input_shape_info}
        self.task2decomposer = task2decomposer
        self.task2n_agents = task2n_agents
        self.args = args

        self.c = args.c_step
        self.skill_dim = args.skill_dim

        self.q = Qnet(args)
        self.value = ValueNet(task2input_shape_info, task2decomposer, task2n_agents, decomposer, args)
        self.encoder = Encoder(args)
        self.decoder = Decoder(task2input_shape_info, task2decomposer, task2n_agents, decomposer, args)
        self.planner = PlannerModel(task2input_shape_info, task2decomposer, task2n_agents, decomposer, args)
        self.reward_transformer = Transformer(args.entity_embed_dim, args.head, args.depth, args.entity_embed_dim)
        self.discr = Discriminator(task2input_shape_info, task2decomposer, task2n_agents, decomposer, args)
        self.reward_predict_net = nn.Sequential(nn.Linear(self.args.entity_embed_dim, 128),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(128, 1))
        self.last_out_h = None
        self.last_h_plan = None

        self.coordination = []
        self.specific = []
        self.c_tmp, self.s_tmp = [], []
        self.saved = False

    def init_hidden(self):
        # make hidden states on the same device as model
        return (self.encoder.q_skill.weight.new(1, self.args.entity_embed_dim).zero_(),
                self.encoder.q_skill.weight.new(1, self.args.entity_embed_dim).zero_(),
                self.encoder.q_skill.weight.new(1, self.args.entity_embed_dim).zero_(),
                self.encoder.q_skill.weight.new(1, self.args.entity_embed_dim).zero_(),
                self.encoder.q_skill.weight.new(1, self.args.entity_embed_dim).zero_())

    def forward_seq_action(self, seq_inputs, hidden_state_dec, hidden_state_plan, task, mask=False, t=0, actions=None):
        seq_act = []
        for i in range(self.c):
            act, hidden_state_dec, hidden_state_plan = self.forward_action(
                seq_inputs[:, i, :], hidden_state_dec, hidden_state_plan, task,  mask, t, actions[:, i])
            if i == 0:
                hidden_state = hidden_state_dec
                h_plan = hidden_state_plan
            seq_act.append(act)
        seq_act = th.stack(seq_act, dim=1)

        return seq_act, hidden_state, h_plan

    def forward_action(self, inputs, emb_inputs, discr_h, hidden_state_dec, hidden_state_plan, task,
                       mask=False, t=0, actions=None):
        h_plan = hidden_state_plan
        act, h_dec, cls_out = self.decoder(emb_inputs, inputs, discr_h, hidden_state_dec, task, mask, actions)
        return act, h_dec, h_plan, cls_out

    def forward_value(self, inputs, hidden_state_value, task, actions=None):
        attn_out, hidden_state_value = self.value(inputs, hidden_state_value, task)
        return attn_out, hidden_state_value

    def forward_value_skill(self, inputs, hidden_state_value, task):
        total_hidden = th.cat(
            [inputs, hidden_state_value.reshape(-1, 1, self.args.entity_embed_dim)], dim=1)
        attn_out, hidden_state_value = self.value.predict(total_hidden)
        return attn_out, hidden_state_value

    # TODO: 放到gumble-muzero中时就要看这里，是否
    def forward_planner(self, inputs, hidden_state_plan, t, task,
                        actions=None, next_inputs=None, loss_out=False, skill_index_out=False):
        # 始终获取所有可能的返回值
        out_h, h, obs_loss, skill_index = self.planner(inputs, hidden_state_plan, t, task,
                                          next_inputs=next_inputs, actions=actions, loss_out=loss_out, 
                                          skill_index_out=skill_index_out)
        
        # 根据参数设置返回值
        if not skill_index_out:
            skill_index = None
            
        return out_h, h, obs_loss, skill_index

    def forward_planner_feedforward(self, emb_inputs, forward_type='action'):
        out_h = self.planner.feedforward(emb_inputs, forward_type)
        return out_h

    def forward_discriminator(self, inputs, t, task, hidden_state_dis):
        dis_out, dis_out_h, h_dis = self.discr(inputs, t, task, hidden_state_dis)
        return dis_out, dis_out_h, h_dis

    def forward_contrastive(self, inputs, inputs_pos):
        logits = self.discr.compute_logits(inputs, inputs_pos)
        return logits

    def forward(self, inputs, hidden_state_plan, hidden_state_dec, hidden_state_dis, t, task, skill,
                mask=False, actions=None, local_obs=None, test_mode=None, skill_index_out=False):
        # TODO: 这里要进行修改，将skill在forward函数中得到并传出到外面。并且只有在t%c=0的时候才计算skill
        if t % self.c == 0:
            # h_plan是hidden_state_plan
            out_h, h_plan, _, skill_index = self.forward_planner(inputs, hidden_state_plan, t, task, skill_index_out)
            # 上一行是得到skill的表示，这一行是将skill融合得到真正的action code。在下面通过decoder解码成单独的action
            out_h = self.forward_planner_feedforward(out_h)
            # TODO: 将skill维护在动作选择里面，就不用在外面显示保存skill了。我是要将这一步skill的选择替换为MCTS
            self.last_out_h, self.last_h_plan = out_h, h_plan
        _, discr_h, h_dis = self.forward_discriminator(inputs, t, task, hidden_state_dis)
        discr_h  = discr_h.reshape(-1, 1, self.args.entity_embed_dim)
        act, h_dec, _ = self.decoder(self.last_out_h, inputs, discr_h, hidden_state_dec, task, mask, actions)
        if skill_index_out == False:
            return act, self.last_h_plan, h_dec, h_dis
        else:
            return act, self.last_h_plan, h_dec, h_dis, skill_index
    def forward_action_skill(self, inputs, hidden_state_dec, hidden_state_dis, t, task, skill, 
                                 mask=False, actions=None):
        # 这个参数t其实没用上，在forward_discriminator里没有使用
        _, discr_h, h_dis = self.forward_discriminator(inputs, t, task, hidden_state_dis)
        discr_h  = discr_h.reshape(-1, 1, self.args.entity_embed_dim)
        act, h_dec, _ = self.decoder(skill, inputs, discr_h, hidden_state_dec, task, mask, actions)
        return act, h_dec, h_dis
    
    # TODO:测试一下reward是否输出形状合适
    def forward_reward_skill(self, inputs, hidden_state_reward, task=None):
        total_hidden = th.cat(
            [inputs, hidden_state_reward.reshape(-1, 1, self.args.entity_embed_dim)], dim=1)
        outputs = self.reward_transformer(total_hidden, None)
        h = outputs[:, -1:, :]
        # 0 应该是代表own。1:enemy是enemy的，1+enemy+ally是ally的
        reward = outputs[:, 0, :]
        reward = self.reward_predict_net(reward)
        return reward, h

    def get_codebook(self):
        """返回agent中planner的skill模块的codebook"""
        if hasattr(self.planner, 'skill_module') and hasattr(self.planner.skill_module, 'emb'):
            return self.planner.skill_module.emb.weight
        return None

    def get_total_agents(self, task):
        """返回特定任务的总agent数目（ally + enemy）"""
        if task in self.task2decomposer:
            task_decomposer = self.task2decomposer[task]
            return task_decomposer.n_agents + task_decomposer.n_enemies
        return 0


class StateEncoder(nn.Module):
    def __init__(self, task2input_shape_info, task2decomposer, task2n_agents, decomposer, args):
        super(StateEncoder, self).__init__()

        self.task2last_action_shape = {task: task2input_shape_info[task]["last_action_shape"] for task in
            task2input_shape_info}
        self.task2decomposer = task2decomposer
        for key in task2decomposer.keys():
            task2decomposer_ = task2decomposer[key]
            break

        self.task2n_agents = task2n_agents
        self.args = args

        self.skill_dim = args.skill_dim

        self.embed_dim = args.mixing_embed_dim
        self.attn_embed_dim = args.attn_embed_dim
        self.entity_embed_dim = args.entity_embed_dim

        # get detailed state shape information
        state_nf_al, state_nf_en, timestep_state_dim = \
        task2decomposer_.state_nf_al, task2decomposer_.state_nf_en, task2decomposer_.timestep_number_state_dim
        self.state_last_action, self.state_timestep_number = task2decomposer_.state_last_action, task2decomposer_.state_timestep_number

        self.n_actions_no_attack = task2decomposer_.n_actions_no_attack

        # define state information processor
        if self.state_last_action:
            self.ally_encoder = nn.Linear(state_nf_al + (self.n_actions_no_attack + 1) * 2, self.entity_embed_dim)
            self.enemy_encoder = nn.Linear(state_nf_en + 1, self.entity_embed_dim)
        else:
            self.ally_encoder = nn.Linear(state_nf_al + (self.n_actions_no_attack + 1), self.entity_embed_dim)
            self.enemy_encoder = nn.Linear(state_nf_en + 1, self.entity_embed_dim)

        # we ought to do attention
        self.query = nn.Linear(self.entity_embed_dim, self.attn_embed_dim)
        self.key = nn.Linear(self.entity_embed_dim, self.attn_embed_dim)

        self.ln = nn.LayerNorm(self.entity_embed_dim)
        self.ally_to_ally = nn.Linear(self.entity_embed_dim*2, self.entity_embed_dim)
        self.ally_to_enemy = nn.Linear(self.entity_embed_dim*2, self.entity_embed_dim)

    def forward(self, states, hidden_state, task, actions=None):
        states = states.unsqueeze(1)

        task_decomposer = self.task2decomposer[task]
        task_n_agents = self.task2n_agents[task]
        last_action_shape = self.task2last_action_shape[task]

        bs = states.size(0)
        n_agents = task_decomposer.n_agents
        n_enemies = task_decomposer.n_enemies
        n_entities = n_agents + n_enemies

        # get decomposed state information
        ally_states, enemy_states, last_action_states, timestep_number_state = task_decomposer.decompose_state(states)
        ally_states = th.stack(ally_states, dim=0)  # [n_agents, bs, 1, state_nf_al]

        _, current_attack_action_info, current_compact_action_states = task_decomposer.decompose_action_info(
            F.one_hot(actions.reshape(-1), num_classes=self.task2last_action_shape[task]))
        current_compact_action_states = current_compact_action_states.reshape(bs, n_agents, -1).permute(1, 0, 2).unsqueeze(2)
        ally_states = th.cat([ally_states, current_compact_action_states], dim=-1)

        current_attack_action_info = current_attack_action_info.reshape(bs, n_agents, n_enemies).sum(dim=1)
        attack_action_states = (current_attack_action_info > 0).type(ally_states.dtype).reshape(bs, n_enemies, 1, 1).permute(1, 0, 2, 3)
        enemy_states = th.stack(enemy_states, dim=0)  # [n_enemies, bs, 1, state_nf_en]
        enemy_states = th.cat([enemy_states, attack_action_states], dim=-1)

        # stack action information
        if self.state_last_action:
            last_action_states = th.stack(last_action_states, dim=0)
            _, _, compact_action_states = task_decomposer.decompose_action_info(last_action_states)
            ally_states = th.cat([ally_states, compact_action_states], dim=-1)

        # do inference and get entity_embed
        ally_embed = self.ally_encoder(ally_states)
        enemy_embed = self.enemy_encoder(enemy_states)

        # we ought to do self-attention
        entity_embed = th.cat([ally_embed, enemy_embed], dim=0)

        # do attention
        proj_query = self.query(entity_embed).permute(1, 2, 0, 3).reshape(bs, n_entities, self.attn_embed_dim)
        proj_key = self.key(entity_embed).permute(1, 2, 3, 0).reshape(bs, self.attn_embed_dim, n_entities)
        energy = th.bmm(proj_query / (self.attn_embed_dim ** (1 / 2)), proj_key)
        attn_score = F.softmax(energy, dim=1)
        proj_value = entity_embed.permute(1, 2, 3, 0).reshape(bs, self.entity_embed_dim, n_entities)
        attn_out = th.bmm(proj_value, attn_score).squeeze(1).permute(0, 2, 1)

        attn_out = attn_out[:, :n_agents].reshape(bs, n_agents, self.entity_embed_dim)
        return attn_out, hidden_state


class ObsEncoder(nn.Module):
    """  sotax agent for multi-task learning """
    def __init__(self, task2input_shape_info, task2decomposer, task2n_agents, decomposer, args):
        super(ObsEncoder, self).__init__()
        self.task2last_action_shape = {task: task2input_shape_info[task]["last_action_shape"] for task in
            task2input_shape_info}
        self.task2decomposer = task2decomposer
        self.task2n_agents = task2n_agents
        self.args = args

        self.skill_dim = args.skill_dim

        self.entity_embed_dim = args.entity_embed_dim
        self.attn_embed_dim = args.attn_embed_dim
        obs_own_dim = decomposer.own_obs_dim
        obs_en_dim, obs_al_dim = decomposer.obs_nf_en, decomposer.obs_nf_al
        n_actions_no_attack = decomposer.n_actions_no_attack
        ## get wrapped obs_own_dim
        wrapped_obs_own_dim = obs_own_dim + args.id_length + n_actions_no_attack + 1
        ## enemy_obs ought to add attack_action_info
        obs_en_dim += 1

        self.ally_value = nn.Linear(obs_al_dim, self.entity_embed_dim)
        self.enemy_value = nn.Linear(obs_en_dim, self.entity_embed_dim)
        self.own_value = nn.Linear(wrapped_obs_own_dim, self.entity_embed_dim)

        self.transformer = Transformer(self.entity_embed_dim, args.head, args.depth, self.entity_embed_dim)

    def forward(self):
        return


class ValueNet(nn.Module):
    def __init__(self, task2input_shape_info, task2decomposer, task2n_agents, decomposer, args):
        super(ValueNet, self).__init__()
        self.task2last_action_shape = {task: task2input_shape_info[task]["last_action_shape"] for task in
            task2input_shape_info}
        self.task2decomposer = task2decomposer
        self.task2n_agents = task2n_agents
        self.args = args

        self.skill_dim = args.skill_dim

        self.entity_embed_dim = args.entity_embed_dim
        self.attn_embed_dim = args.attn_embed_dim
        obs_own_dim = decomposer.own_obs_dim
        obs_en_dim, obs_al_dim = decomposer.obs_nf_en, decomposer.obs_nf_al
        n_actions_no_attack = decomposer.n_actions_no_attack
        ## get wrapped obs_own_dim
        wrapped_obs_own_dim = obs_own_dim + args.id_length + n_actions_no_attack + 1
        ## enemy_obs ought to add attack_action_info
        obs_en_dim += 1

        self.ally_value = nn.Linear(obs_al_dim, self.entity_embed_dim)
        self.enemy_value = nn.Linear(obs_en_dim, self.entity_embed_dim)
        self.own_value = nn.Linear(wrapped_obs_own_dim, self.entity_embed_dim)

        self.ln = nn.Sequential(nn.LayerNorm(self.entity_embed_dim), nn.Tanh())
        self.transformer = Transformer(self.entity_embed_dim, args.head, args.depth, self.entity_embed_dim)

        self.q_skill = nn.Linear(self.entity_embed_dim, self.skill_dim)
        self.reward_fc = nn.Sequential(nn.Linear(self.entity_embed_dim, 128),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(128, 1))

    def init_hidden(self):
        # make hidden states on the same device as model
        return self.q_skill.weight.new(1, self.entity_embed_dim).zero_()

    def encode(self, inputs, hidden_state, task):
        hidden_state = hidden_state.reshape(-1, 1, self.entity_embed_dim)
        # get decomposer, last_action_shape and n_agents of this specific task
        task_decomposer = self.task2decomposer[task]
        task_n_agents = self.task2n_agents[task]
        last_action_shape = self.task2last_action_shape[task]

        # decompose inputs into observation inputs, last_action_info, agent_id_info
        obs_dim = task_decomposer.obs_dim
        obs_inputs, last_action_inputs, agent_id_inputs = inputs[:, :obs_dim], \
        inputs[:, obs_dim:obs_dim + last_action_shape], inputs[:,
        obs_dim + last_action_shape:]

        # decompose observation input
        own_obs, enemy_feats, ally_feats = task_decomposer.decompose_obs(
            obs_inputs)  # own_obs: [bs*self.n_agents, own_obs_dim]
        bs = int(own_obs.shape[0] / task_n_agents)

        # embed agent_id inputs and decompose last_action_inputs
        agent_id_inputs = [
            th.as_tensor(binary_embed(i + 1, self.args.id_length, self.args.max_agent), dtype=own_obs.dtype) for i in
            range(task_n_agents)]
        agent_id_inputs = th.stack(agent_id_inputs, dim=0).repeat(bs, 1).to(own_obs.device)
        _, attack_action_info, compact_action_states = task_decomposer.decompose_action_info(last_action_inputs)

        # incorporate agent_id embed and compact_action_states
        own_obs = th.cat([own_obs, agent_id_inputs, compact_action_states], dim=-1)

        # incorporate attack_action_info into enemy_feats
        attack_action_info = attack_action_info.transpose(0, 1).unsqueeze(-1)
        enemy_feats = th.cat([th.stack(enemy_feats, dim=0), attack_action_info], dim=-1)
        ally_feats = th.stack(ally_feats, dim=0)

        # compute key, query and value for attention
        own_hidden = self.own_value(own_obs).unsqueeze(1)
        ally_hidden = self.ally_value(ally_feats).permute(1, 0, 2)
        enemy_hidden = self.enemy_value(enemy_feats).permute(1, 0, 2)
        history_hidden = hidden_state

        total_hidden = th.cat([own_hidden, enemy_hidden, ally_hidden, history_hidden], dim=1)
        return total_hidden

    def encode_for_skill(self, inputs, hidden_state, task):
        own_obs, enemy_feats, ally_feats = inputs

        # compute key, query and value for attention
        own_hidden = self.own_value(own_obs).unsqueeze(1)
        ally_hidden = self.ally_value(ally_feats)
        enemy_hidden = self.enemy_value(enemy_feats)
        history_hidden = hidden_state

        total_hidden = th.cat([own_hidden, enemy_hidden, ally_hidden, history_hidden], dim=1)
        return total_hidden

    def predict(self, total_hidden):
        outputs = self.transformer(total_hidden, None)
        h = outputs[:, -1:, :]
        reward = outputs[:, 0, :]
        reward = self.reward_fc(reward)
        return reward, h

    def forward(self, inputs, hidden_state, task):
        total_hidden = self.encode(inputs, hidden_state, task)
        reward, h = self.predict(total_hidden)
        return reward, h

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.args = args

        self.skill_dim = args.skill_dim
        self.entity_embed_dim = args.entity_embed_dim

        self.q_skill = nn.Linear(self.entity_embed_dim, self.skill_dim)

    def forward(self, attn_out):
        skill = self.q_skill(attn_out)
        return skill


class BasicDecoder(nn.Module):

    def __init__(self, task2input_shape_info, task2decomposer, task2n_agents, decomposer, args):
        super(BasicDecoder, self).__init__()
        self.task2last_action_shape = {task: task2input_shape_info[task]["last_action_shape"] for task in
            task2input_shape_info}
        self.task2decomposer = task2decomposer
        self.task2n_agents = task2n_agents
        self.args = args

        self.skill_dim = args.skill_dim

        #### define various dimension information
        ## set attributes
        self.entity_embed_dim = args.entity_embed_dim
        self.attn_embed_dim = args.attn_embed_dim
        ## get obs shape information
        obs_own_dim = decomposer.own_obs_dim
        obs_en_dim, obs_al_dim = decomposer.obs_nf_en, decomposer.obs_nf_al
        n_actions_no_attack = decomposer.n_actions_no_attack
        ## get wrapped obs_own_dim
        wrapped_obs_own_dim = obs_own_dim + args.id_length + n_actions_no_attack + 1
        ## enemy_obs ought to add attack_action_info
        obs_en_dim += 1

        self.transformer = Transformer(self.entity_embed_dim, args.head, args.depth, self.entity_embed_dim)
        self.base_q_skill = nn.Linear(self.entity_embed_dim, n_actions_no_attack)
        self.ally_q_skill = nn.Linear(self.entity_embed_dim, 1)

    def init_hidden(self):
        # make hidden states on the same device as model
        return self.q_skill.weight.new(1, self.args.entity_embed_dim).zero_()

    def forward(self, emb_inputs, inputs, hidden_state, task, mask=False, actions=None):
        hidden_state = hidden_state.reshape(-1, 1, self.entity_embed_dim)
        n_agents = self.task2n_agents[task]
        bs, n_agents, n_entity, _ = inputs.shape
        n_enemy = n_entity - n_agents

        total_hidden = emb_inputs.reshape(bs * n_agents, -1, self.entity_embed_dim)
        outputs = self.transformer(total_hidden, None)

        h = outputs[:, -1, :]
        base_action_inputs = outputs[:, 0]

        q_base = self.base_q_skill(base_action_inputs)
        attack_action_inputs = outputs[:, 1:n_enemy+1]
        q_attack = self.ally_q_skill(attack_action_inputs)
        q = th.cat([q_base, q_attack.reshape(-1, n_enemy)], dim=-1)

        return q, h


class Decoder(nn.Module):
    """  sotax agent for multi-task learning """

    def __init__(self, task2input_shape_info, task2decomposer, task2n_agents, decomposer, args):
        super(Decoder, self).__init__()
        self.task2last_action_shape = {task: task2input_shape_info[task]["last_action_shape"] for task in
            task2input_shape_info}
        self.task2decomposer = task2decomposer
        self.task2n_agents = task2n_agents
        self.args = args

        self.skill_dim = args.skill_dim
        self.cls_dim = 3

        #### define various dimension information
        ## set attributes
        self.entity_embed_dim = args.entity_embed_dim
        self.attn_embed_dim = args.attn_embed_dim
        ## get obs shape information
        obs_own_dim = decomposer.own_obs_dim
        obs_en_dim, obs_al_dim = decomposer.obs_nf_en, decomposer.obs_nf_al
        n_actions_no_attack = decomposer.n_actions_no_attack
        ## get wrapped obs_own_dim
        wrapped_obs_own_dim = obs_own_dim + args.id_length + n_actions_no_attack + 1
        ## enemy_obs ought to add attack_action_info
        obs_en_dim += 1

        self.ally_value = nn.Linear(obs_al_dim, self.entity_embed_dim)
        self.enemy_value = nn.Linear(obs_en_dim, self.entity_embed_dim)
        self.own_value = nn.Linear(wrapped_obs_own_dim, self.entity_embed_dim)

        self.transformer = Transformer(self.entity_embed_dim, args.head, args.depth, self.entity_embed_dim)

        self.skill_enc = nn.Linear(self.skill_dim, self.entity_embed_dim)
        self.q_skill = nn.Linear(self.entity_embed_dim * 2, n_actions_no_attack)
        self.base_q_skill = MLPNet(self.entity_embed_dim*2, n_actions_no_attack, 128, output_norm=False)
        self.ally_q_skill = MLPNet(self.entity_embed_dim*2, 1, 128, output_norm=False)

        self.n_actions_no_attack = n_actions_no_attack
        self.cls_hidden = nn.Parameter(th.zeros(1, 1, self.entity_embed_dim))
        self.cls_fc = nn.Linear(self.entity_embed_dim, self.cls_dim)
        self.cross_attn = CrossAttention(task2input_shape_info, task2decomposer, task2n_agents, decomposer, args)

    def init_hidden(self):
        # make hidden states on the same device as model
        return self.q_skill.weight.new(1, self.args.entity_embed_dim).zero_()

    def forward(self, emb_inputs, inputs, discr_h, hidden_state, task, mask=False, actions=None):
        hidden_state = hidden_state.reshape(-1, 1, self.entity_embed_dim)
        cls_hidden = discr_h

        # get decomposer, last_action_shape and n_agents of this specific task
        task_decomposer = self.task2decomposer[task]
        task_n_agents = self.task2n_agents[task]
        last_action_shape = self.task2last_action_shape[task]

        # decompose inputs into observation inputs, last_action_info, agent_id_info
        obs_dim = task_decomposer.obs_dim
        obs_inputs, last_action_inputs, agent_id_inputs = inputs[:, :obs_dim], \
        inputs[:, obs_dim:obs_dim + last_action_shape], \
        inputs[:, obs_dim + last_action_shape:]

        # decompose observation input
        own_obs, enemy_feats, ally_feats = task_decomposer.decompose_obs(
            obs_inputs)  # own_obs: [bs*self.n_agents, own_obs_dim]
        bs = int(own_obs.shape[0] / task_n_agents)

        # embed agent_id inputs and decompose last_action_inputs
        agent_id_inputs = [
            th.as_tensor(binary_embed(i + 1, self.args.id_length, self.args.max_agent), dtype=own_obs.dtype) for i in
            range(task_n_agents)]
        agent_id_inputs = th.stack(agent_id_inputs, dim=0).repeat(bs, 1).to(own_obs.device)
        _, attack_action_info, compact_action_states = task_decomposer.decompose_action_info(last_action_inputs)

        # incorporate agent_id embed and compact_action_states
        own_obs = th.cat([own_obs, agent_id_inputs, compact_action_states], dim=-1)

        # incorporate attack_action_info into enemy_feats
        attack_action_info = attack_action_info.transpose(0, 1).unsqueeze(-1)
        enemy_feats = th.cat([th.stack(enemy_feats, dim=0), attack_action_info], dim=-1)
        ally_feats = th.stack(ally_feats, dim=0)

        enemy_feats = enemy_feats.permute(1, 0, 2)
        ally_feats = ally_feats.permute(1, 0, 2)
        n_enemy, n_ally = enemy_feats.shape[1], ally_feats.shape[1]
        n_entity = n_enemy + n_ally + 1

        # random mask
        if mask and actions is not None:
            actions = actions.reshape(-1)

            b, n, _ = enemy_feats.shape
            mask = th.randint(0, 2, (b, n, 1)).to(enemy_feats.device)
            for i in range(actions.shape[0]):
                if actions[i] > self.n_actions_no_attack-1:
                    mask[i, actions[i]-self.n_actions_no_attack] = 1
            enemy_feats = enemy_feats * mask

            b, n, _ = ally_feats.shape
            mask = th.randint(0, 2, (b, n, 1)).to(ally_feats.device)
            ally_feats = ally_feats * mask

        # compute key, query and value for attention
        own_hidden = self.own_value(own_obs).unsqueeze(1)
        ally_hidden = self.ally_value(ally_feats)
        enemy_hidden = self.enemy_value(enemy_feats)
        history_hidden = hidden_state
        own_emb_inputs, enemy_emb_inputs, ally_emb_inputs = emb_inputs
        emb_hidden = th.cat([own_emb_inputs, enemy_emb_inputs, ally_emb_inputs], dim=1)
        total_hidden = th.cat([own_hidden, enemy_hidden, ally_hidden, emb_hidden, history_hidden], dim=1)

        outputs = self.transformer(total_hidden, None)
        h = outputs[:, -1, :]
        outputs = outputs[:, : n_entity]

        cls_out = self.cls_fc(th.zeros_like(h).detach())
        skill_hidden = discr_h.reshape(-1, 1, self.entity_embed_dim).repeat(1, outputs.shape[1], 1)
        outputs = th.cat([outputs, skill_hidden], dim=-1)
        base_action_inputs = outputs[:, 0, :]
        q_base = self.base_q_skill(base_action_inputs)
        attack_action_inputs = outputs[:, 1: 1+n_enemy]
        q_attack = self.ally_q_skill(attack_action_inputs)
        q = th.cat([q_base, q_attack.reshape(-1, n_enemy)], dim=-1)

        return q, h, cls_out


class Qnet(nn.Module):

    def __init__(self, args):
        super(Qnet, self).__init__()
        self.args = args

        self.skill_dim = args.skill_dim
        self.entity_embed_dim = args.entity_embed_dim

        self.q_skill = nn.Linear(self.entity_embed_dim*2, self.skill_dim)
        self.attack_q_skill = nn.Linear(self.entity_embed_dim*2, 1)

    def forward(self, inputs):
        q = self.q_skill(inputs)

        return q


class PlannerModel(nn.Module):
    """  dynamics model for multi-task learning """

    def __init__(self, task2input_shape_info, task2decomposer, task2n_agents, decomposer, args):
        super(PlannerModel, self).__init__()
        self.task2last_action_shape = {task: task2input_shape_info[task]["last_action_shape"] for task in
            task2input_shape_info}
        self.task2decomposer = task2decomposer
        self.task2n_agents = task2n_agents
        self.args = args

        self.skill_dim = args.skill_dim
        self.vq_skill = args.vq_skill

        #### define various dimension information
        ## set attributes
        self.entity_embed_dim = args.entity_embed_dim
        self.attn_embed_dim = args.attn_embed_dim
        ## get obs shape information
        obs_own_dim = decomposer.own_obs_dim
        obs_en_dim, obs_al_dim = decomposer.obs_nf_en, decomposer.obs_nf_al
        n_actions_no_attack = decomposer.n_actions_no_attack
        ## get wrapped obs_own_dim
        wrapped_obs_own_dim = obs_own_dim + args.id_length + n_actions_no_attack + 1
        ## enemy_obs ought to add attack_action_info
        obs_en_dim += 1

        self.ally_value = nn.Linear(obs_al_dim, self.entity_embed_dim)
        self.enemy_value = nn.Linear(obs_en_dim, self.entity_embed_dim)
        self.own_value = nn.Linear(wrapped_obs_own_dim, self.entity_embed_dim)
        self.value_vale = nn.Linear(1, self.entity_embed_dim)
        self.transformer = Transformer(self.entity_embed_dim, args.head, args.depth, self.entity_embed_dim)
        self.obs_decoder = Transformer(self.entity_embed_dim, args.head, args.depth, self.entity_embed_dim)

        self.base_q_skill = nn.Linear(self.entity_embed_dim * 2, n_actions_no_attack)
        self.ally_q_skill = nn.Linear(self.entity_embed_dim * 2, 1)

        self.own_fc = MLPNet(self.entity_embed_dim, wrapped_obs_own_dim, 128, 3, False)
        self.enemy_fc = MLPNet(self.entity_embed_dim, obs_en_dim, 128, 3, False)
        self.ally_fc = MLPNet(self.entity_embed_dim, obs_al_dim, 128, 3, False)

        self.ln = nn.Sequential(nn.LayerNorm(self.entity_embed_dim), nn.Tanh())

        self.act_own_forward = MLPNet(self.entity_embed_dim, self.entity_embed_dim, 128)
        self.act_enemy_forward = MLPNet(self.entity_embed_dim, self.entity_embed_dim, 128)
        self.act_ally_forward = MLPNet(self.entity_embed_dim, self.entity_embed_dim, 128)

        self.value_own_forward = MLPNet(self.entity_embed_dim, self.entity_embed_dim, 128)
        self.value_enemy_forward = MLPNet(self.entity_embed_dim, self.entity_embed_dim, 128)
        self.value_ally_forward = MLPNet(self.entity_embed_dim, self.entity_embed_dim, 128)

        self.rew_own_forward = MLPNet(self.entity_embed_dim, self.entity_embed_dim, 128)
        self.rew_enemy_forward = MLPNet(self.entity_embed_dim, self.entity_embed_dim, 128)
        self.rew_ally_forward = MLPNet(self.entity_embed_dim, self.entity_embed_dim, 128)

        self.n_actions_no_attack = n_actions_no_attack
        self.reset_last()
        self.skill_module = SkillModule(args)
        self.rec_module = MergeRec(task2input_shape_info, task2decomposer, task2n_agents, decomposer, args)

    def init_hidden(self):
        # make hidden states on the same device as model
        return self.base_q_skill.weight.new(1, self.args.entity_embed_dim).zero_()

    def reset_last(self):
        self.last_own = None
        self.last_enemy = None
        self.last_ally = None

    def add_last(self, own, enemy, ally):
        self.last_own = own
        self.last_enemy = enemy
        self.last_ally = ally

    def feedforward(self, inputs, forward_type='action'):
        assert forward_type in ['action', 'value', 'reward']
        own_emb, enemy_emb, ally_emb = inputs
        n_enemy, n_ally = enemy_emb.shape[1], ally_emb.shape[1]
        if forward_type == 'action':
            own_out = self.act_own_forward(own_emb)
            enemy_out = self.act_enemy_forward(enemy_emb)
            ally_out = self.act_ally_forward(ally_emb)
        elif forward_type == 'value':
            own_out = self.value_own_forward(own_emb)
            enemy_out = self.value_enemy_forward(enemy_emb)
            ally_out = self.value_ally_forward(ally_emb)
        elif forward_type == 'reward':
            own_out = self.rew_own_forward(own_emb)
            enemy_out = self.rew_enemy_forward(enemy_emb)
            ally_out = self.rew_ally_forward(ally_emb)

        return [own_out, enemy_out, ally_out]
    # inputs就是obs+last_action+agent_id
    # next_inputs在原文中就是states，没有更改过。这里rec_module做的应该是根据skill和obs去重建未来的states
    def forward(self, inputs, hidden_state, t, task,
                test=True, next_inputs=None, actions=None, loss_out=False, skill_index_out=False):
        hidden_state = hidden_state.reshape(-1, 1, self.entity_embed_dim)
        # get decomposer, last_action_shape and n_agents of this specific task
        task_decomposer = self.task2decomposer[task]
        task_n_agents = self.task2n_agents[task]
        last_action_shape = self.task2last_action_shape[task]

        # decompose inputs into observation inputs, last_action_info, agent_id_info
        obs_dim = task_decomposer.obs_dim
        obs_inputs, last_action_inputs, agent_id_inputs = inputs[:, :obs_dim], \
        inputs[:, obs_dim:obs_dim + last_action_shape], \
        inputs[:, obs_dim + last_action_shape:]

        # decompose observation input
        # enemy_feats是一个list，长度为enemy_num, 包含了所有敌方智能体的特征，ally_feats也是一个list，包含了所有友方智能体的特征
        own_obs, enemy_feats, ally_feats = task_decomposer.decompose_obs(
            obs_inputs)  # own_obs: [bs*self.n_agents, own_obs_dim]
        bs = int(own_obs.shape[0] / task_n_agents)

        # embed agent_id inputs and decompose last_action_inputs
        agent_id_inputs = [
            th.as_tensor(binary_embed(i + 1, self.args.id_length, self.args.max_agent), dtype=own_obs.dtype) for i in
            range(task_n_agents)]
        agent_id_inputs = th.stack(agent_id_inputs, dim=0).repeat(bs, 1).to(own_obs.device)
        _, attack_action_info, compact_action_states = task_decomposer.decompose_action_info(last_action_inputs)

        # incorporate agent_id embed and compact_action_states
        own_obs = th.cat([own_obs, agent_id_inputs, compact_action_states], dim=-1)

        # incorporate attack_action_info into enemy_feats
        attack_action_info = attack_action_info.transpose(0, 1).unsqueeze(-1)
        # e.g.,原来enemy_feats为list，长度为enemy_num, 每一个元素为[bs * n_agents, obs_en_dim]，现在cat后就变成了[enemy_num, bs * n_agents, obs_en_dim+1]了
        enemy_feats = th.cat([th.stack(enemy_feats, dim=0), attack_action_info], dim=-1)
        ally_feats = th.stack(ally_feats, dim=0)
        # batch, n_enemy, n_feats
        enemy_feats = enemy_feats.permute(1, 0, 2)
        # batch, n_ally, n_feats. 注意ally是比己方智能体数目少1的，因为还有一个是own
        ally_feats = ally_feats.permute(1, 0, 2)
        n_enemy, n_ally = enemy_feats.shape[1], ally_feats.shape[1]

        own_stack, enemy_stack, ally_stack = own_obs.unsqueeze(1).unsqueeze(1), enemy_feats.unsqueeze(1), \
        ally_feats.unsqueeze(1)

        # compute key, query and value for attention
        own_hidden = self.own_value(own_stack)
        ally_hidden = self.ally_value(ally_stack)
        enemy_hidden = self.enemy_value(enemy_stack)
        history_hidden = hidden_state.unsqueeze(1)

        b = own_hidden.shape[0]
        total_hidden = th.cat([own_hidden, enemy_hidden, ally_hidden, history_hidden], dim=2)
        total_hidden = total_hidden.reshape(b, -1, self.entity_embed_dim)

        outputs = self.transformer(total_hidden, None).reshape(b, self.args.num_stack_frames, -1, self.entity_embed_dim)
        h = outputs[:, -1, -1]
        outputs = outputs[:, :, :-1]

        commit_loss = th.tensor(0.).to(inputs.device)
        if self.vq_skill:
            outputs, skill_index, commit_loss = self.skill_module(outputs)
        else:
            skill_index = None  # 非VQ模式时返回None

        own_out_h = outputs[:, -1, 0].unsqueeze(1)
        enemy_out_h = outputs[:, -1, 1:1+n_enemy]
        ally_out_h = outputs[:, -1, 1+n_enemy:1+n_enemy+n_ally]

        own_out, enemy_out, ally_out = own_out_h, enemy_out_h, ally_out_h

        out_loss = th.tensor(0.).to(inputs.device)
        
        if next_inputs is not None and loss_out:
            out_loss = self.rec_module([own_out, enemy_out, ally_out], next_inputs, task,
                                t=t, actions=actions)
            out_loss += commit_loss
        
        return [own_out_h, enemy_out_h, ally_out_h], h, out_loss, skill_index


class Discriminator(nn.Module):
    def __init__(self, task2input_shape_info, task2decomposer, task2n_agents, decomposer, args):
        super(Discriminator, self).__init__()
        self.task2last_action_shape = {task: task2input_shape_info[task]["last_action_shape"] for task in
            task2input_shape_info}
        self.task2decomposer = task2decomposer
        self.task2n_agents = task2n_agents
        self.args = args

        self.skill_dim = args.skill_dim
        self.ssl_type = args.ssl_type

        #### define various dimension information
        ## set attributes
        self.entity_embed_dim = args.entity_embed_dim
        self.attn_embed_dim = args.attn_embed_dim
        ## get obs shape information
        obs_own_dim = decomposer.own_obs_dim
        obs_en_dim, obs_al_dim = decomposer.obs_nf_en, decomposer.obs_nf_al
        n_actions_no_attack = decomposer.n_actions_no_attack
        ## get wrapped obs_own_dim
        wrapped_obs_own_dim = obs_own_dim + args.id_length + n_actions_no_attack + 1
        ## enemy_obs ought to add attack_action_info
        obs_en_dim += 1

        self.ally_value = nn.Linear(obs_al_dim, self.entity_embed_dim)
        self.enemy_value = nn.Linear(obs_en_dim, self.entity_embed_dim)
        self.own_value = nn.Linear(wrapped_obs_own_dim, self.entity_embed_dim)
        self.transformer = Transformer(self.entity_embed_dim, args.head, args.depth, self.entity_embed_dim)
        self.W = nn.Parameter(th.rand(self.entity_embed_dim, self.entity_embed_dim))

        if args.ssl_type == 'moco':
            self.act_proj = nn.Sequential(nn.Linear(self.entity_embed_dim, 128),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(128, self.entity_embed_dim),
                                          nn.LayerNorm(self.entity_embed_dim), nn.Tanh())
            self.ssl_proj = nn.Sequential(nn.Linear(self.entity_embed_dim, 128),
                                          #   nn.BatchNorm1d(128),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(128, self.entity_embed_dim))
        elif args.ssl_type == 'byol':
            self.act_proj = nn.Sequential(nn.Linear(self.entity_embed_dim, 128),
                                          nn.BatchNorm1d(128),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(128, self.entity_embed_dim))
            self.ssl_proj = nn.Sequential(nn.Linear(self.entity_embed_dim, 128),
                                          nn.BatchNorm1d(128),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(128, self.entity_embed_dim))

    def forward(self, inputs, t, task, hidden_state):
        # get decomposer, last_action_shape and n_agents of this specific task
        task_decomposer = self.task2decomposer[task]
        task_n_agents = self.task2n_agents[task]
        last_action_shape = self.task2last_action_shape[task]

        # decompose inputs into observation inputs, last_action_info, agent_id_info
        obs_dim = task_decomposer.obs_dim
        obs_inputs, last_action_inputs, agent_id_inputs = inputs[:, :obs_dim], \
        inputs[:, obs_dim:obs_dim + last_action_shape], \
        inputs[:, obs_dim + last_action_shape:]

        # decompose observation input
        own_obs, enemy_feats, ally_feats = task_decomposer.decompose_obs(
            obs_inputs)  # own_obs: [bs*self.n_agents, own_obs_dim]
        bs = int(own_obs.shape[0] / task_n_agents)

        # embed agent_id inputs and decompose last_action_inputs
        agent_id_inputs = [
            th.as_tensor(binary_embed(i + 1, self.args.id_length, self.args.max_agent), dtype=own_obs.dtype) for i in
            range(task_n_agents)]
        agent_id_inputs = th.stack(agent_id_inputs, dim=0).repeat(bs, 1).to(own_obs.device)
        _, attack_action_info, compact_action_states = task_decomposer.decompose_action_info(last_action_inputs)

        # incorporate agent_id embed and compact_action_states
        own_obs = th.cat([own_obs, agent_id_inputs, compact_action_states], dim=-1)

        # incorporate attack_action_info into enemy_feats
        attack_action_info = attack_action_info.transpose(0, 1).unsqueeze(-1)
        enemy_feats = th.cat([th.stack(enemy_feats, dim=0), attack_action_info], dim=-1)
        ally_feats = th.stack(ally_feats, dim=0)

        enemy_feats = enemy_feats.permute(1, 0, 2)
        ally_feats = ally_feats.permute(1, 0, 2)

        # compute key, query and value for attention
        own_hidden = self.own_value(own_obs).unsqueeze(1)
        ally_hidden = self.ally_value(ally_feats)
        enemy_hidden = self.enemy_value(enemy_feats)
        history_hidden = hidden_state

        b = own_hidden.shape[0]
        total_hidden = th.cat([own_hidden, enemy_hidden, ally_hidden], dim=1)
        outputs = self.transformer(total_hidden, None)
        h = history_hidden

        own_out_h = outputs[:, 0].reshape(-1, self.entity_embed_dim)
        own_out = self.ssl_proj(own_out_h)
        if self.ssl_type == 'moco':
            own_out_h = self.act_proj(own_out_h)
        elif self.ssl_type == 'byol':
            own_out_h = self.act_proj(own_out)

        return own_out, own_out_h, h

    def compute_logits(self, z, z_pos):
        Wz = th.matmul(self.W, z_pos.T)  # (z_dim,B)
        logits = th.matmul(z, Wz)  # (B,B)
        logits = logits - th.max(logits, 1)[0][:, None]
        return logits


class CrossAttention(nn.Module):
    def __init__(self, task2input_shape_info, task2decomposer, task2n_agents, decomposer, args):
        super(CrossAttention, self).__init__()

        self.task2last_action_shape = {task: task2input_shape_info[task]["last_action_shape"] for task in
            task2input_shape_info}
        self.task2decomposer = task2decomposer
        for key in task2decomposer.keys():
            task2decomposer_ = task2decomposer[key]
            break

        self.task2n_agents = task2n_agents
        self.args = args

        self.skill_dim = args.skill_dim

        self.embed_dim = args.mixing_embed_dim
        self.attn_embed_dim = args.attn_embed_dim
        self.entity_embed_dim = args.entity_embed_dim

        # get detailed state shape information
        state_nf_al, state_nf_en, timestep_state_dim = \
        task2decomposer_.state_nf_al, task2decomposer_.state_nf_en, task2decomposer_.timestep_number_state_dim
        self.state_last_action, self.state_timestep_number = task2decomposer_.state_last_action, task2decomposer_.state_timestep_number

        self.n_actions_no_attack = task2decomposer_.n_actions_no_attack

        # define state information processor
        if self.state_last_action:
            self.ally_encoder = nn.Linear(state_nf_al + (self.n_actions_no_attack + 1) * 2, self.entity_embed_dim)
            self.enemy_encoder = nn.Linear(state_nf_en + 1, self.entity_embed_dim)
        else:
            self.ally_encoder = nn.Linear(state_nf_al + (self.n_actions_no_attack + 1), self.entity_embed_dim)
            self.enemy_encoder = nn.Linear(state_nf_en + 1, self.entity_embed_dim)

        # we ought to do attention
        self.query = nn.Linear(self.entity_embed_dim, self.attn_embed_dim)
        self.key = nn.Linear(self.entity_embed_dim, self.attn_embed_dim)

    def forward(self, dec_emb, skill_emb, task, actions=None):
        skill_emb = th.cat(skill_emb, dim=1)
        dec_emb, skill_emb = dec_emb.unsqueeze(1), skill_emb.unsqueeze(1)

        task_decomposer = self.task2decomposer[task]
        task_n_agents = self.task2n_agents[task]
        last_action_shape = self.task2last_action_shape[task]

        n_agents = task_decomposer.n_agents
        n_enemies = task_decomposer.n_enemies
        n_entities = n_agents + n_enemies
        bs = dec_emb.shape[0]

        # do attention
        proj_query = self.query(dec_emb).reshape(bs, n_entities, self.attn_embed_dim)
        proj_key = self.key(skill_emb).permute(0, 1, 3, 2).reshape(bs, self.attn_embed_dim, n_entities)
        energy = th.bmm(proj_query / (self.attn_embed_dim ** (1 / 2)), proj_key)
        attn_score = F.softmax(energy, dim=1)
        proj_value = dec_emb.permute(0, 1, 3, 2).reshape(bs, self.entity_embed_dim, n_entities)
        attn_out = th.bmm(proj_value, attn_score).squeeze(1).permute(0, 2, 1)

        attn_out = attn_out.reshape(bs, n_entities, self.entity_embed_dim)
        return attn_out


class MergeRec(nn.Module):
    def __init__(self, task2input_shape_info, task2decomposer, task2n_agents, decomposer, args):
        super(MergeRec, self).__init__()
        self.task2last_action_shape = {task: task2input_shape_info[task]["last_action_shape"] for task in
            task2input_shape_info}
        self.task2decomposer = task2decomposer
        for key in task2decomposer.keys():
            task2decomposer_ = task2decomposer[key]
            break

        self.task2n_agents = task2n_agents
        self.args = args

        self.skill_dim = args.skill_dim

        self.embed_dim = args.mixing_embed_dim
        self.attn_embed_dim = args.attn_embed_dim
        self.entity_embed_dim = args.entity_embed_dim

        # get detailed state shape information
        state_nf_al, state_nf_en, timestep_state_dim = \
        task2decomposer_.state_nf_al, task2decomposer_.state_nf_en, task2decomposer_.timestep_number_state_dim
        self.state_last_action, self.state_timestep_number = task2decomposer_.state_last_action, task2decomposer_.state_timestep_number

        self.n_actions_no_attack = task2decomposer_.n_actions_no_attack

        # define state information processor
        if self.state_last_action:
            self.ally_encoder = nn.Linear(state_nf_al + (self.n_actions_no_attack + 1) * 2, self.entity_embed_dim)
            self.enemy_encoder = nn.Linear(state_nf_en + 1, self.entity_embed_dim)
        else:
            self.ally_encoder = nn.Linear(state_nf_al + (self.n_actions_no_attack + 1), self.entity_embed_dim)
            self.enemy_encoder = nn.Linear(state_nf_en + 1, self.entity_embed_dim)

        # we ought to do attention
        self.own_qk = nn.Linear(self.entity_embed_dim, self.attn_embed_dim*2)
        self.enemy_qk = nn.Linear(self.entity_embed_dim, self.attn_embed_dim*2)
        self.enemy_ref_qk = nn.Linear(self.entity_embed_dim, self.attn_embed_dim*2)
        self.norm = nn.Sequential(nn.LayerNorm(self.attn_embed_dim), nn.Tanh())

        self.enemy_hidden = nn.Parameter(th.zeros(1, 1, self.entity_embed_dim)).requires_grad_(True)
        self.last_enemy_h = None
        if self.state_last_action:
            self.ally_dec_fc = MLPNet(self.entity_embed_dim, state_nf_al + (self.n_actions_no_attack + 1) * 2, 128)
            self.enemy_dec_fc = MLPNet(self.entity_embed_dim, state_nf_en + 1, 128)
        else:
            self.ally_dec_fc = MLPNet(self.entity_embed_dim, state_nf_al + (self.n_actions_no_attack + 1), 128)
            self.enemy_dec_fc = MLPNet(self.entity_embed_dim, state_nf_en + 1, 128)
            
        # 添加新的观察预测网络
        self.obs_pred = MLPNet(self.entity_embed_dim + decomposer.n_enemies * self.entity_embed_dim, 
                               decomposer.obs_dim, 128)

    def global_process(self, states, task, actions=None):
        states = states.unsqueeze(1)

        task_decomposer = self.task2decomposer[task]
        task_n_agents = self.task2n_agents[task]
        last_action_shape = self.task2last_action_shape[task]

        bs = states.size(0)
        n_agents = task_decomposer.n_agents
        n_enemies = task_decomposer.n_enemies
        n_entities = n_agents + n_enemies

        # get decomposed state information
        ally_states, enemy_states, last_action_states, timestep_number_state = task_decomposer.decompose_state(states)
        ally_states = th.stack(ally_states, dim=0)  # [n_agents, bs, 1, state_nf_al]

        _, current_attack_action_info, current_compact_action_states = task_decomposer.decompose_action_info(
            F.one_hot(actions.reshape(-1), num_classes=self.task2last_action_shape[task]))
        current_compact_action_states = current_compact_action_states.reshape(bs, n_agents, -1).permute(1, 0, 2).unsqueeze(2)
        ally_states = th.cat([ally_states, current_compact_action_states], dim=-1)

        current_attack_action_info = current_attack_action_info.reshape(bs, n_agents, n_enemies).sum(dim=1)
        attack_action_states = (current_attack_action_info > 0).type(ally_states.dtype).reshape(
            bs, n_enemies, 1, 1).permute(1, 0, 2, 3)
        enemy_states = th.stack(enemy_states, dim=0)  # [n_enemies, bs, 1, state_nf_en]
        enemy_states = th.cat([enemy_states, attack_action_states], dim=-1)

        # stack action information
        if self.state_last_action:
            last_action_states = th.stack(last_action_states, dim=0)
            _, _, compact_action_states = task_decomposer.decompose_action_info(last_action_states)
            ally_states = th.cat([ally_states, compact_action_states], dim=-1)

        ally_states = ally_states.permute(1, 2, 0, 3).reshape(bs, n_agents, -1)
        enemy_states = enemy_states.permute(1, 2, 0, 3).reshape(bs, n_enemies, -1)
        return [ally_states, enemy_states]

    def attn_process(self, emb_inputs, emb_q, emb_k):
        # do attention
        bs, n, _ = emb_inputs.shape
        proj_query = emb_q
        proj_key = emb_k.permute(0, 2, 1)
        energy = th.bmm(proj_query / (self.attn_embed_dim ** (1 / 2)), proj_key)
        attn_score = F.softmax(energy, dim=1)
        proj_value = emb_inputs.permute(0, 2, 1)
        attn_out = th.bmm(proj_value, attn_score).permute(0, 2, 1)
        attn_out = attn_out.reshape(bs, n, self.entity_embed_dim)

        return attn_out
    def pred_next_obs(self, emb_inputs, task, t=0, actions=None):
        own_emb, enemy_emb, ally_emb = emb_inputs
        task_decomposer = self.task2decomposer[task]
        task_n_agents = self.task2n_agents[task]
        n_agents = task_decomposer.n_agents
        n_enemies = task_decomposer.n_enemies
        
        bs = own_emb.shape[0] // n_agents  # 计算批次大小
        
        if t==0:
            self.last_enemy_h = self.enemy_hidden.repeat(bs, n_enemies, 1).unsqueeze(-2)
        # 为什么不使用ally, 是因为own_emb和ally_emb是重合的！ally不就是每一个own的组合吗
        own_emb = own_emb.reshape(bs, n_agents, self.entity_embed_dim)
        enemy_emb = enemy_emb.reshape(bs, n_agents, n_enemies, self.entity_embed_dim).permute(
            0, 2, 1, 3).reshape(-1, n_agents, self.entity_embed_dim)
        enemy_emb = th.cat(
            [self.last_enemy_h.reshape(-1, 1, self.entity_embed_dim), enemy_emb], dim=-2)

        own_q, own_k = self.own_qk(own_emb).chunk(2, -1)
        enemy_q, enemy_k = self.enemy_qk(enemy_emb).chunk(2, -1)

        enemy_ref = self.attn_process(enemy_emb, enemy_q, enemy_k)[:, 0].reshape(
            bs, n_enemies, self.entity_embed_dim)
        enemy_q, enemy_k = self.enemy_ref_qk(enemy_ref).chunk(2, -1)

        total_emb = th.cat([own_emb, enemy_ref], dim=-2)
        total_q = th.cat([own_q, enemy_q], dim=-2)
        total_k = th.cat([own_k, enemy_k], dim=-2)
        total_out = self.attn_process(total_emb, total_q, total_k)

        ally_out = total_out[:, :n_agents]  # [bs, n_agents, entity_embed_dim]
        enemy_out = total_out[:, -n_enemies:]  # [bs, n_enemies, entity_embed_dim]
        self.last_enemy_h = enemy_out

        # 根据要求拼接 ally_out 和 enemy_out：每个 ally 附加所有 enemy 的特征
        # 将 enemy_out 扩展为 [bs, 1, n_enemies, entity_embed_dim]
        enemy_out_expanded = enemy_out.unsqueeze(1)
        
        # 广播并展平为 [bs, n_agents, n_enemies * entity_embed_dim]
        enemy_out_flat = enemy_out_expanded.expand(bs, n_agents, n_enemies, self.entity_embed_dim).reshape(
            bs, n_agents, n_enemies * self.entity_embed_dim)
        
        # 拼接 ally_out 和展平后的 enemy_out
        agent_out = th.cat([ally_out, enemy_out_flat], dim=-1)  # [bs, n_agents, entity_embed_dim + n_enemies * entity_embed_dim]
        
        # 使用 obs_pred 网络预测观察值
        # TODO: 最后一步，怎么把obs_pred的输入大小和输出大小固定住？转向single_task？这个结束之后，设计一个world model的函数（与该函数大部分类似其实，就是不计算loss，就可以了）
        obs_pred_out = self.obs_pred(agent_out).reshape(-1, task_decomposer.obs_dim)
        return obs_pred_out
    def forward(self, emb_inputs, obs, task, t=0, actions=None):
        obs_pred_out = self.pred_next_obs(emb_inputs, task, t=t, actions=actions)
        task_decomposer = self.task2decomposer[task]
        # 计算预测观察值与真实观察值之间的损失
        loss = F.mse_loss(obs_pred_out, obs.reshape(-1, task_decomposer.obs_dim).detach())
        return loss
