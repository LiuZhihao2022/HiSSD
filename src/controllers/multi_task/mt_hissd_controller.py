from re import L
from modules.agents.multi_task import REGISTRY as agent_REGISTRY
from modules.decomposers import REGISTRY as decomposer_REGISTRY

from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
import torch.distributions as D
import numpy as np
import torch.nn.functional as F
import copy
from modules.mixers.multi_task.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.multi_task.qattn import QMixer as MTAttnQMixer

from utils.embed import binary_embed
# This multi-agent controller shares parameters between agents
class HISSDSMAC:
    def __init__(self, train_tasks, task2scheme, task2args, main_args):
        # set some task-specific attributes
        self.train_tasks = train_tasks
        self.task2scheme = task2scheme
        self.task2args = task2args
        self.task2n_agents = {
            task: self.task2args[task].n_agents for task in train_tasks
        }
        self.main_args = main_args

        # set some common attributes
        self.agent_output_type = main_args.agent_output_type
        self.action_selector = action_REGISTRY[main_args.action_selector](main_args)

        # get decomposer for each task
        env2decomposer = {
            "sc2": "sc2_decomposer",
        }
        self.task2decomposer, self.task2dynamic_decoder = {}, {}
        self.surrogate_decomposer = None
        for task in train_tasks:
            task_args = self.task2args[task]
            if task_args.env == "sc2":
                task_decomposer = decomposer_REGISTRY[env2decomposer[task_args.env]](
                    task_args
                )
                self.task2decomposer[task] = task_decomposer
                if not self.surrogate_decomposer:
                    self.surrogate_decomposer = task_decomposer
            else:
                raise NotImplementedError(f"Unsupported env decomposer {task_args.env}")
            # set obs_shape
            task_args.obs_shape = task_decomposer.obs_dim

        # build agents
        # get input dimensions for each task, can choose to append action shape and id shape or not
        task2input_shape_info = self._get_input_shape()
        self._build_agents(task2input_shape_info)

        self.skill_dim = main_args.skill_dim
        self.c_step = main_args.c_step
        self.device = 'cpu'
        
        # 添加mixer的初始化代码
        self.mixer = None
        if main_args.mixer is not None:
            if main_args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif main_args.mixer == "mt_qattn":
                self.mixer = MTAttnQMixer(self.surrogate_decomposer, main_args)
            else:
                raise ValueError(f"Mixer {main_args.mixer} not recognised.")
            self.target_mixer = copy.deepcopy(self.mixer)
            
        self.init_params()

    def init_params(self):
        self.hidden_states_value = None
        self.hidden_states_reward = None
        self.hidden_states_enc = None
        self.hidden_states_dec = None
        self.hidden_states_plan = None
        self.skill_hidden = None
        self.q_skill = None
        self.skill = None
        self.cls_dim = 3
        self.last_out_h = None
        self.last_obs_loss = None
        self.last_skill_index = None
        self.last_pred_states = None

    def select_actions(
        self, ep_batch, t_ep, t_env, task, bs=slice(None), test_mode=False
    ):
        # Only select actions for the selected batch elements in bs
        # 这里是得到具体的action，所以这里要使用ep_batch["avail_actions"]得到具体的action值，而不是avail_skills
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, task, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(
            agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode
        )
        return chosen_actions

    def forward_global_hidden(self, ep_batch, t, task, actions=None, test_mode=False):
        agent_inputs = ep_batch["state"][:, t]
        agent_outs, self.hidden_states_enc = self.agent.forward_global_hidden(
            agent_inputs, task, self.hidden_states_enc, actions=actions
        )

        return agent_outs.reshape(ep_batch.batch_size * self.task2n_agents[task], 1, -1)
    # 这个obs_emb应该是skill，名字有误导
    def forward_global_action(
        self, ep_batch, obs_emb, disrc_h, t, task, test_model=False
    ):
        bs = ep_batch.batch_size
        agent_inputs = self._build_inputs(ep_batch, t, task)
        # disrc_h = disrc_h.reshape(-1, 1, self.main_args.entity_embed_dim)
        act_out, self.hidden_states_dec, self.hidden_states_plan = (
            self.agent.forward_action(
                agent_inputs,
                obs_emb,
                # disrc_h,
                None,
                self.hidden_states_dec,
                self.hidden_states_plan,
                task,
                t=t,
            )
        )
        # return act_out.reshape(bs, 1, self.task2n_agents[task], -1), cls_out.reshape(
        #     bs, 1, self.task2n_agents[task], self.cls_dim
        # )
        return act_out.reshape(bs, 1, self.task2n_agents[task], -1)

    def forward_value(self, ep_batch, t, task, test_mode=False, actions=None):
        bs = ep_batch.batch_size
        agent_inputs = self._build_inputs(ep_batch, t, task)
        agent_outs, self.hidden_states_value = self.agent.forward_value(
            agent_inputs, self.hidden_states_value, task, actions=actions
        )

        return agent_outs.reshape(bs, self.task2n_agents[task], 1)

    def forward_value_skill(self, bs, batch_emb, task):
        batch_emb = th.cat(batch_emb, dim=1)
        agent_outs, self.hidden_states_value = self.agent.forward_value_skill(
            batch_emb, self.hidden_states_value, task
        )

        return agent_outs.reshape(bs, self.task2n_agents[task], 1)

    def forward_seq_action(self, ep_batch, t, task, mask=False, test_model=False):
        agent_seq_inputs = []
        for i in range(self.c_step):
            agent_inputs = self._build_inputs(ep_batch, t + i, task)
            agent_seq_inputs.append(agent_inputs)
        agent_seq_inputs = th.stack(agent_seq_inputs, dim=1)
        actions = ep_batch["actions"][:, t : t + self.c_step]

        agent_seq_outs, self.hidden_states_dec, self.hidden_states_plan = (
            self.agent.forward_seq_action(
                agent_seq_inputs,
                self.hidden_states_dec,
                self.hidden_states_plan,
                task,
                mask,
                t,
                actions,
            )
        )

        return agent_seq_outs.view(
            ep_batch.batch_size, self.c_step, self.task2n_agents[task], -1
        )

    def forward_reward_skill(self, bs, batch_emb, task):
        batch_emb = th.cat(batch_emb, dim=1)
        agent_outs, self.hidden_states_reward = self.agent.forward_reward_skill(
            batch_emb, self.hidden_states_reward, task
        )
        # dim of reward predicted is [bs, 1]
        agent_outs = agent_outs.reshape(bs, self.task2n_agents[task], 1).sum(dim = 1)
        return agent_outs

    def forward_planner(
        self,
        ep_batch,
        t,
        task,
        actions=None,
        test_mode=False,
        training=False,
        hrl=False, # TODO:这个参数在训练VAE的时候一定要开启，不然每次都会进行skill的选择
        loss_out=False,
        skill_index_out=False, # 参数，控制是否返回skill_index
    ):
        # 移除了return_pred参数及相关逻辑
        if t % self.c_step == 0 or hrl == False:
            # agent_inputs -> (bs*n_agents, input_shape)
            agent_inputs = self._build_inputs(ep_batch, t, task)
            states = ep_batch["state"][:, t]
            next_inputs = None
            next_states = None
            if training:  # 只在训练模式下获取next_inputs用于计算损失
                # if t + self.c_step < ep_batch["obs"].shape[1]:  # 确保不会越界
                next_inputs = ep_batch["obs"][:, t + self.c_step]
                next_states = ep_batch["state"][:, t + self.c_step]
                
            # 修改调用方式
            out_h, self.hidden_states_plan, obs_loss, skill_index = self.agent.forward_planner(
                agent_inputs,
                states,
                t,
                task,
                hidden_state_plan=self.hidden_states_plan,
                actions=actions,
                next_inputs=next_inputs,
                next_states=next_states,
                loss_out=loss_out,
                skill_index_out=skill_index_out
            )
            
            # 保存结果
            self.last_out_h, self.last_obs_loss = out_h, obs_loss
            self.last_skill_index = skill_index

        return self.last_out_h, self.last_obs_loss, self.last_skill_index
    # additional_input可以根据forward_type选择不同的输入数据
    # 暂时都使用state作为additional_input
    def forward_planner_feedforward(self, emb_inputs, additional_input=None, forward_type="action",task=None):
        out_h = self.agent.forward_planner_feedforward(emb_inputs, forward_type=forward_type, additional_input=additional_input, task=task)
        return out_h

    def forward_discriminator(self, ep_batch, t, task, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t, task)
        ssl_out, ssl_out_h, self.hidden_states_dis = self.agent.forward_discriminator(
            agent_inputs, t, task, self.hidden_states_dis
        )
        return ssl_out.reshape(
            ep_batch.batch_size, self.task2n_agents[task], -1
        ), ssl_out_h.reshape(ep_batch.batch_size, self.task2n_agents[task], -1)
    # 这个是为了计算contrasive_loss，也就是learner里面的ssl_loss的。single_task用不上
    def forward_contrastive(self, emb_inputs, emb_pos):
        logits = self.agent.forward_contrastive(emb_inputs, emb_pos)
        return logits

    def forward(self, ep_batch, t, task, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t, task)
        # 这里是通过skill得到具体的action，所以这里要使用ep_batch["avail_actions"]得到具体的action值，而不是avail_skills
        avail_actions = ep_batch["avail_actions"][:, t]
        actions = ep_batch["actions"][:, t]

        bs = agent_inputs.shape[0] // self.task2n_agents[task]
        # 看上去好像是有时间上抽象的? c_step个时间重新选一次skill
        # 留出了接口，但是在默认配置里c_step = 1，即每一个时间步都选择一次skill. To be implement
        # TODO:但是在agent的实现里，skill就是直接进行了一个赋值？那这个skill应该是不能用的
        if t % self.c_step == 0:
            (
                agent_outs,
                self.hidden_states_plan,
                self.hidden_states_dec,
                self.hidden_states_dis,
                self.skill,
            ) = self.agent(
                agent_inputs,
                self.hidden_states_dec,
                self.hidden_states_dis,
                t,
                task,
                None,
                hidden_state_plan=self.hidden_states_plan,
                actions=actions,
                local_obs=None,
                test_mode=test_mode,
            )
        else:
            (
                agent_outs,
                self.hidden_states_plan,
                self.hidden_states_dec,
                self.hidden_states_dis,
                _,
            ) = self.agent(
                agent_inputs,
                self.hidden_states_dec,
                self.hidden_states_dis,
                t,
                task,
                self.skill,
                hidden_state_plan=self.hidden_states_plan,
                actions=actions,
                local_obs=None,
                test_mode=test_mode,
            )

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.main_args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(
                    ep_batch.batch_size * self.task2n_agents[task], -1
                )
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

            if not test_mode and self.main_args.adaptation:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.main_args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(
                        dim=1, keepdim=True
                    ).float()

                agent_outs = (
                    1 - self.action_selector.epsilon
                ) * agent_outs + th.ones_like(
                    agent_outs
                ) * self.action_selector.epsilon / epsilon_action_num

                if getattr(self.main_args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        return agent_outs.view(ep_batch.batch_size, self.task2n_agents[task], -1)

    def init_hidden(self, batch_size, task):
        # we always know we are in which task when do init_hidden
        n_agents = self.task2n_agents[task]
        (
            hidden_states_value,
            hidden_states_reward,
            hidden_states_dec,
            hidden_states_plan,
            hidden_states_dis,
        ) = self.agent.init_hidden()
        self.hidden_states_value = hidden_states_value.unsqueeze(0).expand(
            batch_size, n_agents, -1
        )
        self.hidden_states_reward = hidden_states_reward.unsqueeze(0).expand(
            batch_size, n_agents, -1
        )
        self.hidden_states_dec = hidden_states_dec.unsqueeze(0).expand(
            batch_size, n_agents, -1
        )
        self.hidden_states_plan = hidden_states_plan.unsqueeze(0).expand(
            batch_size, n_agents, -1
        )
        self.hidden_states_dis = hidden_states_dis.unsqueeze(0).expand(
            batch_size, n_agents, -1
        )
    # 不仅是initialize world model,还是forward action with skill 的 latent
    # TODO: 应该两个初始化的时机都是一样的，区别是wm latent需要存在tree buffer里，action的不用，存在self里即可
    def init_hidden_wm(self, batch_size, task):
        """
        Initializes the hidden states for the world model (WM) of the agents 
        for a specific task.

        Args:
            batch_size (int): The number of samples in the batch.
            task (str): The task identifier used to determine the number of agents.

        Returns:
            list: A list containing the initialized hidden states for reward and value 
                  networks, each with dimensions expanded to match the batch size 
                  and number of agents.
        """
        n_agents = self.task2n_agents[task]
        (
            hidden_states_value,
            hidden_states_reward,
            hidden_states_dec_for_act,
            hidden_states_plan,
            hidden_states_dis_for_act,
        ) = self.agent.init_hidden()
        hidden_states_value = hidden_states_value.unsqueeze(0).expand(
            batch_size, n_agents, -1
        )
        hidden_states_reward = hidden_states_reward.unsqueeze(0).expand(
            batch_size, n_agents, -1
        )
        self.hidden_states_dec_for_act = hidden_states_dec_for_act.unsqueeze(0).expand(
            batch_size, n_agents, -1
        )
        self.hidden_states_dis_for_act = hidden_states_dis_for_act.unsqueeze(0).expand(
            batch_size, n_agents, -1
        )
        hidden_states_reward = hidden_states_reward.unsqueeze(1)
        hidden_states_value = hidden_states_value.unsqueeze(1)
        hidden_state_wm = th.cat([hidden_states_reward, hidden_states_value], dim=1)
        return hidden_state_wm

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        """we don't load the state of task dynamic decoder"""
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
        self.device = 'cuda'

    def save_models(self, path):
        """we don't save the state of task dynamic decoder"""
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))
        # 保存mixer参数
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))

    def load_models(self, path):
        """we don't load the state of task_encoder"""
        self.agent.load_state_dict(
            th.load(
                "{}/agent.th".format(path), map_location=lambda storage, loc: storage
            )
        )
        # 加载mixer参数
        if self.mixer is not None:
            self.mixer.load_state_dict(
                th.load(
                    "{}/mixer.th".format(path),
                    map_location=lambda storage, loc: storage,
                )
            )

    def _build_agents(self, task2input_shape_info):
        self.agent = agent_REGISTRY[self.main_args.agent](
            task2input_shape_info,
            self.task2decomposer,
            self.task2n_agents,
            self.surrogate_decomposer,
            self.main_args,
        )

    def _build_actions(self, actions):
        actions = actions.reshape(-1) - 5
        zeros = th.zeros_like(actions).to(self.main_args.device)
        actions = th.where(actions >= 0, actions, zeros)
        return actions

    def _build_inputs(self, batch, t, task):
        """
        Builds the input tensor for the agents at a given time step.
        Args:
            batch (Batch): The batch of data containing observations, actions, etc.
            t (int): The current time step.
            task (str): The task identifier.
        Returns:
            Tensor: The input tensor for the agents at the given time step.
        Notes:
            - Assumes homogenous agents with flat observations.
            - If `obs_last_action` is True in task arguments, includes the last action taken by the agents.
            - If `obs_agent_id` is True in task arguments, includes the agent IDs as a one-hot encoded tensor.
        """
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])
        task_args, n_agents = self.task2args[task], self.task2n_agents[task]
        if task_args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])
        if task_args.obs_agent_id:
            inputs.append(
                th.eye(n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1)
            )

        inputs = th.cat([x.reshape(bs * n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self):
        task2input_shape_info = {}
        for task in self.train_tasks:
            task_scheme = self.task2scheme[task]
            input_shape = task_scheme["obs"]["vshape"]
            last_action_shape, agent_id_shape = 0, 0
            if self.task2args[task].obs_last_action:
                input_shape += task_scheme["actions_onehot"]["vshape"][0]
                last_action_shape = task_scheme["actions_onehot"]["vshape"][0]
            if self.task2args[task].obs_agent_id:
                input_shape += self.task2n_agents[task]
                agent_id_shape = self.task2n_agents[task]
            task2input_shape_info[task] = {
                "input_shape": input_shape,
                "last_action_shape": last_action_shape,
                "agent_id_shape": agent_id_shape,
            }
        return task2input_shape_info
    
    # 这个函数还是需要smac的框架，因为要做online的交互，不过要把ma-gumbel-muzero给引进来
    def forward_action_skill(self, ep_batch, t, skill_index, task, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t, task)
        # 这里是通过skill得到具体的action，所以这里要使用ep_batch["avail_actions"]得到具体的action值，而不是avail_skills
        avail_actions = ep_batch["avail_actions"][:, t]
        
        # 使用skill_index获取对应的code
        device = agent_inputs.device
        # get skill这对吗？是几个skill啊？
        skill_index = np.array(skill_index)
        skill_code = self.get_skill(skill_index)
        
        task_args, n_agents = self.task2args[task], self.task2n_agents[task]
        task_decomposer = self.task2decomposer[task]
        n_enemy = task_decomposer.n_enemies
        n_ally = n_agents - 1
        own_skill = skill_code[:, 0].unsqueeze(1)
        enemy_skill = skill_code[:, 1:1+n_enemy]
        ally_skill = skill_code[:, 1+n_enemy:1+n_enemy+n_ally]
        all_skill = [own_skill, enemy_skill, ally_skill]
        action_out_h = self.forward_planner_feedforward(
            all_skill, additional_input=agent_inputs, forward_type="action", task=task)
        # 这个forward_action_skill是自定义的，专门用于与online interaction时的框架，所以不需要hidden_state_plan，只需要skill就行
        (
            agent_outs,
            _, # 这个_是hidden_states_plan, 这个在这里不需要
            self.hidden_states_dec_for_act,
            self.hidden_states_dis_for_act,
            _
        ) = self.agent(
            agent_inputs,
            self.hidden_states_dec_for_act,
            self.hidden_states_dis_for_act,
            t,
            task,
            action_out_h, # 这个就是skill, 专门为action的
        )
        if self.agent_output_type == "pi_logits":

            if getattr(self.main_args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(
                    ep_batch.batch_size * self.task2n_agents[task], -1
                )
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

            if not test_mode and self.main_args.adaptation:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.main_args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(
                        dim=1, keepdim=True
                    ).float()

                agent_outs = (
                    1 - self.action_selector.epsilon
                ) * agent_outs + th.ones_like(
                    agent_outs
                ) * self.action_selector.epsilon / epsilon_action_num

                if getattr(self.main_args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        return agent_outs.view(ep_batch.batch_size, self.task2n_agents[task], -1)
    def get_codebook(self):
        """返回agent中planner的skill模块的codebook"""
        return self.agent.get_codebook()
    def get_skill(self, skill_index):
        """返回agent中planner的skill模块的skill"""
        return self.agent.get_skill(skill_index)
    def get_total_agents(self, task):
        """返回特定任务的总agent数目（ally + enemy）"""
        return self.agent.get_total_agents(task)

    # 这个函数接口应该契合ma-gumbel-muzero的设计
    # 这里的skill_index已经是转换后的，可以与codebook中skill所对应上的index了，每一个元素值的范围都是[0, skill_dim-1]
    def world_model_predict(self, batch_obs, batch_state, skill_index, hidden_state_wm, task):
        # batch_obs本身就已经包含last action和agent id了
        # 这个batch应该是一个包含batch_size的，但是到mctx里面怎么batch地使用？
        # Convert hidden state w am to numpy array, then to float tensor
        bs, n_agents, _ = batch_obs.shape
        hidden_state_wm = np.array(hidden_state_wm, dtype=np.float32)
        hidden_state_wm = th.tensor(hidden_state_wm, dtype=th.float32, device=self.device)
        hidden_state_reward, hidden_state_value = hidden_state_wm[:, 0], hidden_state_wm[:, 1]

        batch_obs_shape = batch_obs.shape
        batch_state_shape = batch_state.shape
        hidden_state_shape = hidden_state_value.shape

        hidden_state_reward = hidden_state_reward.reshape(bs*n_agents, -1)
        hidden_state_value = hidden_state_value.reshape(bs*n_agents, -1)
        batch_obs = th.FloatTensor(batch_obs).to(self.device).reshape(bs*n_agents, -1)
        batch_state = th.FloatTensor(batch_state).to(self.device).reshape(bs, -1)
        skill_index = th.tensor(skill_index, device=self.device, dtype=th.int64)
        # batch_last_action = th.FloatTensor(batch_last_action).to(self.device)
        # # batch_obs = batch_obs.view(bs* n_nodes * n_agents, batch_obs.shape[-1])
        # # Create one-hot encoding for batch_last_action
        # if batch_last_action is not None:
            # one_hot_code = th.zeros(
            #     # *batch_last_action.shape[:-1], self.skill_dim, device=batch_last_action.device
            #     *batch_last_action.shape, self.skill_dim, device=batch_last_action.device
            # )
            # valid_indices = batch_last_action >= 0
            # one_hot_code[valid_indices] = F.one_hot(
            #     batch_last_action[valid_indices].long(), num_classes=self.skill_dim
            # ).float()
        # else:
        #     one_hot_code = th.zeros(bs * n_nodes, n_agents, self.skill_dim, device=self.device)
        # one_hot_code = one_hot_code.view(-1, self.skill_dim)
        # inputs = []
        # inputs.append(batch_obs)
        task_args, n_agents = self.task2args[task], self.task2n_agents[task]
        task_decomposer = self.task2decomposer[task]
        n_enemy = task_decomposer.n_enemies
        n_ally = n_agents - 1
        # if task_args.obs_last_action:
        #     inputs.append(one_hot_code)
        # if task_args.obs_agent_id:
        #     inputs.append(
        #         th.eye(n_agents, device=self.device).unsqueeze(0).expand(bs * n_nodes, -1, -1)
        #     )
        # # TODO: check shape here
        # agent_inputs = th.cat([x.reshape(bs * n_nodes* n_agents, -1) for x in inputs], dim=1)

        
        # skill在寻找最近的时，会先从竖着的找最近，然后转换为横着的。所以这里每一个横着的就是skill，不用再转换了
        outputs = self.get_skill(skill_index)
        shape_except_last = outputs.shape[:-1]
        skill_code = self.agent.planner.task2unsqueeze_mlp[task](outputs).reshape(bs * n_agents, self.get_total_agents(task), self.main_args.entity_embed_dim)
        # TODO: 这里要从skill_code还原出skill embedding
        # TODO:这个skill应该是什么shape?要经过一次reshape才行，但是应该变为什么样？
        own_skill = skill_code[:, 0].unsqueeze(1)
        enemy_skill = skill_code[:, 1:1+n_enemy]
        ally_skill = skill_code[:, 1+n_enemy:1+n_enemy+n_ally]
        all_skill = [own_skill, enemy_skill, ally_skill]
        with th.no_grad():
            # 使用skill_code修改embedding
            # 这里假设skill_code可以直接用于out_h的计算
            
            # 使用MergeRec的pred_next方法预测下一步观察
            
            # 预测下一步观察
            # batch_obs.reshape(-1, batch_obs_shape[-1])
            next_obs, next_state = self.agent.planner.rec_module.pred_next(
                batch_obs, batch_state, all_skill, task
            )
            one_hot_code = F.one_hot(skill_index.long(), num_classes=self.skill_dim).float()
            agent_id = th.eye(n_agents, device=self.device).unsqueeze(0).expand(bs, -1, -1)
            # next_obs = next_obs.reshape(batch_obs.shape)
            # TODO: 这个state的shape不是完全由obs组成的！需要修改以前的部分以使用obs拼接的state
            # next_state = next_obs.reshape(bs, -1)
            next_obs_input = [next_obs, one_hot_code, agent_id]
            next_obs_input = th.cat([x.reshape(bs * n_agents, -1) for x in next_obs_input], dim=1)
            # 创建一个简单的batch字典来传递数据
            # temp_batch = {
            #     "obs": batch_obs,
            #     "state": batch_state
            # }
            # 使用HISSDAgent的forward_reward_skill预测奖励
            # 为reward预测准备输入
            reward_out_h = self.forward_planner_feedforward(
                all_skill, additional_input=batch_obs, forward_type="reward", task=task
            )
            batch_emb_reward = th.cat(reward_out_h, dim=1)
            # 要手动传入hidden state，所以不用mac里定义的forward_reward_skill
            reward_pred, hidden_state_reward = self.agent.forward_reward_skill(
                batch_emb_reward, hidden_state_reward, task
            )
            reward_pred = reward_pred.reshape(bs, n_agents).sum(dim=1)
            
            # 新增: 使用HISSDAgent的forward_value_skill预测价值
            # 为value预测准备输入
            # TODO: value的预测是使用这个函数还是forward_value函数？
            
            value_out_h = self.forward_planner_feedforward(
                all_skill, additional_input=batch_obs, forward_type="value",task=task
            )
            batch_emb_value = th.cat(value_out_h, dim=1)
            value_pred_pre, hidden_state_value = self.agent.forward_value_skill(
                batch_emb_value, hidden_state_value, task
            )
            value_pred = self.mixer(value_pred_pre.reshape(bs, 1, *value_pred_pre.shape[-2:]), batch_state.reshape(bs , 1 ,batch_state.shape[-1]), self.task2decomposer[task])
            value_pred = value_pred.reshape(bs, )
            # update hidden state
            hidden_state_reward = hidden_state_reward.reshape(hidden_state_shape).unsqueeze(1)
            hidden_state_value = hidden_state_value.reshape(hidden_state_shape).unsqueeze(1)
            new_hidden_state_wm = th.cat([hidden_state_reward, hidden_state_value], dim=1)
            next_obs_input = next_obs_input.reshape(bs, n_agents, -1).detach().cpu().numpy()
            next_state = next_state.reshape(bs, -1).detach().cpu().numpy()
            reward_pred = reward_pred.reshape(bs, ).detach().cpu().numpy()
            value_pred = value_pred.reshape(bs, ).detach().cpu().numpy()
            new_hidden_state_wm = new_hidden_state_wm.detach().cpu().numpy()
            return next_obs_input, next_state, reward_pred, value_pred, new_hidden_state_wm

    def preprocess_obs(self, batch, t, task):
        # inputs = th.cat([x.reshape(bs * n_agents, -1) for x in inputs], dim=1)
        # batch_obs = batch_obs.reshape(-1, batch_obs.shape[-1])
        # 调用agent的预处理函数
        agent_inputs = self._build_inputs(batch, t, task)
        # 暂时不处理直接返回
        # return self.agent.preprocess_obs(agent_inputs, task)
        return agent_inputs.reshape(batch.batch_size, self.task2n_agents[task], -1)
