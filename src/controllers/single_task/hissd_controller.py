import torch as th
import torch.nn.functional as F
import numpy as np
import copy

from modules.agents.single_task import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from modules.mixers.qmix import QMixer

# This multi-agent controller shares parameters between agents
class HISSDSMAC:
    def __init__(self, scheme, args):
        # Set attributes
        self.args = args
        self.n_agents = args.n_agents
        self.device = 'cpu'
        
        # Setup action selection
        self.agent_output_type = args.agent_output_type
        self.action_selector = action_REGISTRY[args.action_selector](args)
        
        # Setup agent parameters
        self.input_shape = self._get_input_shape(scheme)
        self._build_agents()

        self.skill_dim = args.skill_dim
        self.c_step = args.c_step
        
        # Initialize hidden states
        self.hidden_states_value = None
        self.hidden_states_reward = None
        self.hidden_states_dec = None
        self.hidden_states_plan = None
        
        # Setup mixer if specified
        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError(f"Mixer {args.mixer} not recognised.")
            self.target_mixer = copy.deepcopy(self.mixer)
        
        # Initialize skill variables
        self.last_out_h = None
        self.last_skill_index = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs, skill_index = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(
            agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode
        )
        return chosen_actions, skill_index.reshape(ep_batch.batch_size, -1).flatten()

    def forward_value(self, ep_batch, t, test_mode=False, actions=None):
        bs = ep_batch.batch_size
        agent_inputs = self._build_inputs(ep_batch, t)
        agent_outs, self.hidden_states_value = self.agent.forward_value(
            agent_inputs, self.hidden_states_value, actions=actions
        )
        return agent_outs.reshape(bs, self.n_agents, 1)

    def forward_value_skill(self, bs, batch_emb):
        batch_emb = th.cat(batch_emb, dim=1)
        agent_outs, self.hidden_states_value = self.agent.forward_value_skill(
            batch_emb, self.hidden_states_value
        )
        return agent_outs.reshape(bs, self.n_agents, 1)

    def forward_reward_skill(self, bs, batch_emb):
        batch_emb = th.cat(batch_emb, dim=1)
        agent_outs, self.hidden_states_reward = self.agent.forward_reward_skill(
            batch_emb, self.hidden_states_reward
        )
        agent_outs = agent_outs.reshape(bs, self.n_agents, 1).sum(dim=1)
        return agent_outs

    def load_state(self, other_mac):
        """we don't load the state of task dynamic decoder"""
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def forward_planner(
        self,
        ep_batch,
        t,
        actions=None,
        test_mode=False,
        training=True,
        hrl=False,
        loss_out=False,
        skill_index_out=False,
        external_skill_index=None,
    ):
        if t % self.c_step == 0 or hrl == False:
            agent_inputs = self._build_inputs(ep_batch, t)
            states = ep_batch["state"][:, t]
            next_inputs = None
            next_states = None
            
            if training:
                if t + self.c_step < ep_batch["obs"].shape[1]:
                    next_inputs = ep_batch["obs"][:, t + self.c_step]
                    next_states = ep_batch["state"][:, t + self.c_step]
            
            out_h, self.hidden_states_plan, obs_loss, skill_index, loss_dict = self.agent.forward_planner(
                agent_inputs,
                states,
                hidden_state_plan=self.hidden_states_plan,
                actions=actions,
                next_inputs=next_inputs,
                next_states=next_states,
                loss_out=loss_out,
                skill_index_out=skill_index_out,
                training=training,
                external_skill_index=external_skill_index
            )
            
            self.last_out_h, self.last_skill_index = out_h, skill_index

        return self.last_out_h, obs_loss, self.last_skill_index, loss_dict

    def forward_planner_feedforward(self, emb_inputs, additional_input=None, forward_type="action", adaptation=False):
        out_h = self.agent.forward_planner_feedforward(
            emb_inputs, forward_type=forward_type, additional_input=additional_input, adaptation=adaptation
        )
        return out_h

    def forward_contrastive(self, emb_inputs, emb_pos):
        logits = self.agent.forward_contrastive(emb_inputs, emb_pos)
        return logits

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        actions = ep_batch["actions"][:, t]
        states = ep_batch["state"][:, t]
        bs = agent_inputs.shape[0] // self.n_agents
        
        if t % self.c_step == 0:
            (
                agent_outs,
                self.hidden_states_plan,
                self.hidden_states_dec,
                self.skill_index,
            ) = self.agent(
                agent_inputs,
                states,
                self.hidden_states_dec,
                t,
                None,
                hidden_state_plan=self.hidden_states_plan,
                actions=actions,
                skill_index_out=True,
                test_mode=test_mode,
            )
        else:
            (
                agent_outs,
                self.hidden_states_plan,
                self.hidden_states_dec,
                _,
            ) = self.agent(
                agent_inputs,
                states,
                self.hidden_states_dec,
                t,
                None, # TODO: 这里不应该是skill吗？
                hidden_state_plan=self.hidden_states_plan,
                actions=actions,
                skill_index_out=True,
                test_mode=test_mode,
            )

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative
                reshaped_avail_actions = avail_actions.reshape(
                    ep_batch.batch_size * self.n_agents, -1
                )
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

            if not test_mode and self.args.adaptation:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    epsilon_action_num = reshaped_avail_actions.sum(
                        dim=1, keepdim=True
                    ).float()

                agent_outs = (
                    1 - self.action_selector.epsilon
                ) * agent_outs + th.ones_like(
                    agent_outs
                ) * self.action_selector.epsilon / epsilon_action_num

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1), self.skill_index

    def init_hidden(self, batch_size):
        (
            hidden_states_value,
            hidden_states_reward,
            hidden_states_dec,
            hidden_states_plan,
        ) = self.agent.init_hidden()
        
        self.hidden_states_value = hidden_states_value.unsqueeze(0).expand(
            batch_size, self.n_agents, -1
        )
        self.hidden_states_reward = hidden_states_reward.unsqueeze(0).expand(
            batch_size, self.n_agents, -1
        )
        self.hidden_states_dec = hidden_states_dec.unsqueeze(0).expand(
            batch_size, self.n_agents, -1
        )
        self.hidden_states_plan = hidden_states_plan.unsqueeze(0).expand(
            batch_size, self.n_agents, -1
        )

    def parameters(self):
        return self.agent.parameters()

    def cuda(self):
        self.agent.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
        self.device = 'cuda'

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(
            th.load(
                "{}/agent.th".format(path), map_location=lambda storage, loc: storage
            )
        )
        if self.mixer is not None:
            self.mixer.load_state_dict(
                th.load(
                    "{}/mixer.th".format(path),
                    map_location=lambda storage, loc: storage,
                )
            )
            self.target_mixer.load_state_dict(
                th.load(
                    "{}/mixer.th".format(path),
                    map_location=lambda storage, loc: storage,
                )
            )

    def _build_agents(self):
        self.agent = agent_REGISTRY[self.args.agent](
            self.input_shape,
            self.args,
        )

    def _build_inputs(self, batch, t, use_skill=False):
        # Assumes homogenous agents with flat observations.
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])
        
        if self.args.obs_last_action:
            if t == 0:
                if use_skill == False:
                    inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
                else:
                    inputs.append(th.zeros_like(batch["skills_onehot"][:, t]))
            else:
                if use_skill == False:
                    inputs.append(batch["actions_onehot"][:, t - 1])
                else:
                    inputs.append(batch["skills_onehot"][:, (t//self.c_step - 1)*self.c_step])
        
        if self.args.obs_agent_id:
            inputs.append(
                th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1)
            )

        inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape

    def forward_action_skill(self, ep_batch, t, skill_index, bs=None, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        batch_size = ep_batch.batch_size

        device = agent_inputs.device
        skill_index = np.array(skill_index)
        outputs = self.get_skill(skill_index)
        
        action_out_h = self.forward_planner_feedforward(
            outputs,
            additional_input=agent_inputs,
            forward_type="action"
        )

        if t % self.c_step == 0:
            (
                agent_outs,
                _,
                self.hidden_states_dec,
                _
            ) = self.agent(
                agent_inputs,
                None,  # No states needed for action selection
                hidden_state_dec=self.hidden_states_dec,
                t=t,
                skill=action_out_h,
                skill_index_out=False,
                test_mode=True,
            )
        else:
            (
                agent_outs,
                _,
                self.hidden_states_dec,
                _
            ) = self.agent(
                agent_inputs,
                None,  # No states needed for action selection
                hidden_state_dec=self.hidden_states_dec,
                t=t,
                skill=None,
                skill_index_out=False,
                test_mode=True,
            )

        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                reshaped_avail_actions = avail_actions.reshape(
                    batch_size * self.n_agents, -1
                )
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

            if not test_mode and self.args.adaptation:
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    epsilon_action_num = reshaped_avail_actions.sum(
                        dim=1, keepdim=True
                    ).float()
                agent_outs = (
                    1 - self.action_selector.epsilon
                ) * agent_outs + th.ones_like(
                    agent_outs
                ) * self.action_selector.epsilon / epsilon_action_num
                if getattr(self.args, "mask_before_softmax", True):
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        return agent_outs.view(batch_size, self.n_agents, -1)

    def get_codebook(self):
        """返回agent中planner的skill模块的codebook"""
        return self.agent.get_codebook()
        
    def get_skill(self, skill_index):
        """返回agent中planner的skill模块的skill"""
        return self.agent.get_skill(skill_index)

    def world_model_predict(self, batch_obs, batch_state, skill_index, hidden_state_wm):
        # 转换hidden state为可用格式
        bs = batch_obs.shape[0]
        hidden_state_wm = np.array(hidden_state_wm, dtype=np.float32)
        hidden_state_wm = th.tensor(hidden_state_wm, dtype=th.float32, device=self.device)
        hidden_state_reward, hidden_state_value = hidden_state_wm[:, 0], hidden_state_wm[:, 1]

        hidden_state_reward = hidden_state_reward.reshape(bs*self.n_agents, -1)
        hidden_state_value = hidden_state_value.reshape(bs*self.n_agents, -1)
        batch_obs = th.FloatTensor(batch_obs).to(self.device).reshape(bs*self.n_agents, -1)
        batch_state = th.FloatTensor(batch_state).to(self.device).reshape(bs, -1)
        skill_index = th.tensor(skill_index, device=self.device, dtype=th.int64)
        
        # 获取技能表示
        outputs = self.get_skill(skill_index)
        
        with th.no_grad():
            # 使用技能预测下一个观察和状态
            next_obs, next_state = self.agent.planner.rec_module.pred_next(
                batch_obs, batch_state, outputs
            )
            
            # 构建完整输入（包括one-hot skill表示）
            one_hot_code = F.one_hot(skill_index.long(), num_classes=self.skill_dim).float()
            agent_id = th.eye(self.n_agents, device=self.device).unsqueeze(0).expand(bs, -1, -1)
            next_obs_input = [next_obs, one_hot_code, agent_id]
            next_obs_input = th.cat([x.reshape(bs * self.n_agents, -1) for x in next_obs_input], dim=1)
            
            # 预测奖励
            reward_out_h = self.forward_planner_feedforward(
                outputs, additional_input=batch_state, forward_type="reward"
            )
            batch_emb_reward = th.cat(reward_out_h, dim=1)
            reward_pred, hidden_state_reward = self.agent.forward_reward_skill(
                batch_emb_reward, hidden_state_reward
            )
            reward_pred = reward_pred.reshape(bs, self.n_agents).sum(dim=1)
            
            # 预测价值
            value_pred_pre, hidden_state_value = self.agent.forward_value(
                next_obs_input, hidden_state_value
            )
            if self.mixer is not None:
                value_pred = self.mixer(value_pred_pre.reshape(bs, 1, self.n_agents, -1), next_state.reshape(bs, 1, next_state.shape[-1]))
                value_pred = value_pred.reshape(bs, )
            else:
                value_pred = value_pred_pre.reshape(bs, self.n_agents).sum(dim=1)
            
            # 更新hidden state
            hidden_state_reward = hidden_state_reward.reshape(bs, self.n_agents, -1).unsqueeze(1)
            hidden_state_value = hidden_state_value.reshape(bs, self.n_agents, -1).unsqueeze(1)
            new_hidden_state_wm = th.cat([hidden_state_reward, hidden_state_value], dim=1)
            
            # 转换为NumPy数组返回
            next_obs_input = next_obs_input.reshape(bs, self.n_agents, -1).detach().cpu().numpy()
            next_state = next_state.reshape(bs, -1).detach().cpu().numpy()
            reward_pred = reward_pred.reshape(bs, ).detach().cpu().numpy()
            value_pred = value_pred.reshape(bs, ).detach().cpu().numpy()
            new_hidden_state_wm = new_hidden_state_wm.detach().cpu().numpy()
            
            return next_obs_input, next_state, reward_pred, value_pred, new_hidden_state_wm

    def preprocess_obs(self, batch, t, use_skill=False):
        agent_inputs = self._build_inputs(batch, t, use_skill)
        return agent_inputs.reshape(batch.batch_size, self.n_agents, -1)
