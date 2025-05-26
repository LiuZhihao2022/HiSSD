import copy
import random
from sre_compile import dis

from numpy import log
from components.episode_buffer import EpisodeBatch
# 移除mixer导入
import torch as th
from torch.optim import RMSprop, Adam, AdamW
import torch.nn.functional as F
import math
import wandb  # 添加wandb导入

import os


class HISSDLearnerContinueTrain:
    def __init__(self, mac, logger, main_args):
        self.main_args = main_args
        self.mac = mac
        self.logger = logger
        self.use_wandb = main_args.use_wandb if hasattr(main_args, "use_wandb") else False

        # 获取一些属性从mac
        self.task2args = mac.task2args
        self.task2n_agents = mac.task2n_agents
        self.surrogate_decomposer = mac.surrogate_decomposer
        self.task2decomposer = mac.task2decomposer
        self.task2input_shape_info = mac.task2input_shape_info  # 新增：获取输入shape信息
        self.n_actions = {}
        # 使用mac中的mixer而不是在这里初始化。注意，两个mixer在同一个mac中，不在target_mac之中
        self.mixer = mac.mixer
        self.target_mixer = mac.target_mixer

        self.params = list(mac.parameters())
        if self.mixer is not None:
            self.params += list(self.mixer.parameters())

        self.last_target_update_episode = 0

        self._reset_optimizer()

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        # define attributes for each specific task
        self.task2train_info, self.task2encoder_params, self.task2encoder_optimiser = (
            {},
            {},
            {},
        )
        for task in self.task2args:
            task_args = self.task2args[task]
            self.task2train_info[task] = {}
            self.task2train_info[task]["log_stats_t"] = (
                -task_args.learner_log_interval - 1
            )
            self.n_actions[task] = self.task2input_shape_info[task]["last_action_shape"]

        self.c = main_args.c_step
        self.skill_dim = main_args.skill_dim
        self.beta = main_args.beta
        self.alpha = main_args.coef_conservative
        self.phi = main_args.coef_dist
        self.kl_weight = main_args.coef_kl
        self.entity_embed_dim = main_args.entity_embed_dim
        self.device = None
        self.ssl_type = main_args.ssl_type
        self.ssl_tw = main_args.ssl_time_window
        self.double_neg = main_args.double_neg
        self.td_weight = main_args.td_weight
        self.adaptation = main_args.adaptation
        self.epsilon = main_args.epsilon

        self.pretrain_steps = 0
        self.training_steps = 0
        self.reset_last_batch()

        # 新增：InfoNCE投影MLP
        proj_dim = 128
        # 获取obs维度
        input_dim = self.task2input_shape_info[list(self.task2input_shape_info.keys())[0]]["input_shape"]
        # TODO: 因为只有一个task，所以这里直接用task是可以的。不过multi-task的实现中，这样是不可以的
        n_enemy = self.task2decomposer[task].n_enemies
        n_agents = self.task2decomposer[task].n_agents
        self.infonce_proj_skill = th.nn.Sequential(
            th.nn.Linear(self.entity_embed_dim*(n_agents+n_enemy), proj_dim),
            th.nn.ReLU(inplace=True),
            th.nn.Linear(proj_dim, proj_dim)
        )
        self.infonce_proj_act = th.nn.Sequential(
            th.nn.Linear(self.c * self.n_actions[task], proj_dim),
            th.nn.ReLU(inplace=True),
            th.nn.Linear(proj_dim, proj_dim)
        )

    def _reset_optimizer(self):
        if self.main_args.optim_type.lower() == "rmsprop":
            self.pre_optimiser = RMSprop(
                params=self.params,
                lr=self.main_args.lr,
                alpha=self.main_args.optim_alpha,
                eps=self.main_args.optim_eps,
                weight_decay=self.main_args.weight_decay,
            )
            self.optimiser = RMSprop(
                params=self.params,
                lr=self.main_args.lr,
                alpha=self.main_args.optim_alpha,
                eps=self.main_args.optim_eps,
                weight_decay=self.main_args.weight_decay,
            )
        elif self.main_args.optim_type.lower() == "adam":
            self.pre_optimiser = Adam(
                params=self.params,
                lr=self.main_args.lr,
                weight_decay=self.main_args.weight_decay,
            )
            self.optimiser = Adam(
                params=self.params,
                lr=self.main_args.critic_lr,
                weight_decay=self.main_args.weight_decay,
            )
        elif self.main_args.optim_type.lower() == "adamw":
            self.pre_optimiser = AdamW(
                params=self.params,
                lr=self.main_args.lr,
                weight_decay=self.main_args.weight_decay,
            )
            self.optimiser = AdamW(
                params=self.params,
                lr=self.main_args.critic_lr,
                weight_decay=self.main_args.weight_decay,
            )
        else:
            raise ValueError("Invalid optimiser type", self.main_args.optim_type)
        self.pre_optimiser.zero_grad()
        self.optimiser.zero_grad()

    def zero_grad(self):
        self.pre_optimiser.zero_grad()
        self.optimiser.zero_grad()

    def update(self, pretrain=True):
        grad_norm = th.nn.utils.clip_grad_norm_(
            self.params, self.main_args.grad_norm_clip
        )
        if pretrain:
            self.pre_optimiser.step()
            self.pre_optimiser.zero_grad()
        else:
            self.optimiser.step()
            self.optimiser.zero_grad()

    def l2_loss(self, x, y):
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1)

    def contrastive_loss(self, obs, obs_pos, obs_neg):
        # obs & obs_pos: 1, dim; obs_neg: bs, dim
        obs_ = th.cat([obs, obs_neg.detach()], dim=0)
        obs_pos_ = th.cat([obs_pos, obs_neg], dim=0).detach()
        logits = self.mac.forward_contrastive(obs_, obs_pos_)
        labels = th.zeros(logits.shape[0]).long().to(self.device)
        loss = F.cross_entropy(logits, labels)
        return loss

    def reset_last_batch(self):
        self.last_task = ""
        self.last_batch = {}
    # used for self-supervised learning
    def update_last_batch(self, cur_task, cur_batch):
        if not self.double_neg:
            if cur_task != self.last_task:
                self.reset_last_batch()
                self.last_batch[cur_task] = cur_batch
        else:
            self.last_batch[cur_task] = cur_batch
        self.last_task = cur_task

    def compute_neg_sample(self, batch, task):
        target_outs = []
        agent_random = random.randint(0, self.task2n_agents[task] - 1)
        self.target_mac.init_hidden(batch.batch_size, task)
        for t in range(batch.max_seq_length - self.c):
            with th.no_grad():
                target_mac_out, _ = self.target_mac.forward_discriminator(
                    batch, t=t, task=task
                )
                target_mac_out = target_mac_out[:, agent_random]
            target_outs.append(target_mac_out)
        target_outs = th.cat(target_outs, dim=1).reshape(
            -1, self.main_args.entity_embed_dim
        )

        return target_outs
    
    def train_vae(
        self,
        batch: EpisodeBatch,
        t_env: int,
        episode_num: int,
        task: str,
        ssl_loss=None,
    ):
        rewards = batch["reward"][:, :]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :].float()
        mask = batch["filled"][:, :].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        dec_loss = th.tensor(0.0, device = self.device)
        infonce_loss = th.tensor(0.0, device = self.device)
        b, t, n = actions.shape[0], actions.shape[1], actions.shape[2]
        self.mac.init_hidden(batch.batch_size, task)
        t = 0
        while t < batch.max_seq_length - self.c:
            act_outs = []
            agent_inputs = self.mac._build_inputs(batch, t=t, task=task)
            # 得到当前batch的skill embedding和skill index
            agent_outs, _, skill_index, loss_dict = self.mac.forward_planner(
                batch, t=t, task=task, actions=actions[:, t], hrl=True, skill_index_out=True
            )
            act_agent_outs = self.mac.forward_planner_feedforward(
                agent_outs, additional_input=agent_inputs, forward_type="action", task=task
            )
            for i in range(self.c):
                act_out = self.mac.forward_global_action(
                    batch, act_agent_outs, None, t + i, task
                )
                act_outs.append(act_out)
            act_outs = th.stack(act_outs, dim=1)
            _, _, n, a = act_out.shape
            dec_loss += (
                F.cross_entropy(
                    act_outs.reshape(-1, a),
                    actions[:, t : t + self.c].squeeze(-1).reshape(-1),
                    reduction="sum",
                )
                / mask[:, t : t + self.c].sum()
            ) / n

            # ====== InfoNCE部分（skill embedding vs. 动作序列）======
            if self.main_args.compute_infonce:
                skill_dim = self.skill_dim
                bs = batch.batch_size
                n_agents = self.task2n_agents[task]
                device = agent_inputs.device
                total = bs * n_agents
                n_enemy = self.task2decomposer[task].n_enemies
                n_ally = n_agents - 1

                # 1. 构造所有skill-state embedding
                all_skill_embeds = []
                for k in range(skill_dim):
                    skill_idx = th.full((bs, n_agents), k, dtype=th.long, device=device)
                    skill_emb = self.mac.get_skill(skill_idx)  # [bs, n_agents, entity_embed_dim]
                    all_skill_embeds.append(skill_emb.reshape(total, -1))  # [total, entity_embed_dim]
                all_skill_embeds = th.stack(all_skill_embeds, dim=0)  # [skill_dim, total, entity_embed_dim]

                # 获取当前obs（agent_inputs），shape [total, obs_dim]
                obs_dim = self.task2input_shape_info[task]["input_shape"]
                obs_inputs = agent_inputs  # [total, obs_dim]

                # 拼接obs到skill embedding
                obs_inputs_expand = obs_inputs.unsqueeze(0).expand(skill_dim, -1, -1)  # [skill_dim, total, obs_dim]
                obs_inputs_expand = obs_inputs_expand.reshape(skill_dim * total, -1)  # [skill_dim*total, obs_dim]
                # skill_obs_cat = th.cat([all_skill_embeds, obs_inputs_expand], dim=-1)  # [skill_dim, total, entity_embed_dim+obs_dim]

                skill_code = self.mac.agent.planner.task2unsqueeze_mlp[task](all_skill_embeds).reshape(skill_dim*total, self.mac.get_total_agents(task), self.main_args.entity_embed_dim)
                own_skill = skill_code[:, 0].unsqueeze(1)
                enemy_skill = skill_code[:, 1:1+n_enemy]
                ally_skill = skill_code[:, 1+n_enemy:1+n_enemy+n_ally] # TODO: 这里的ally skill已经不对了，这里应该只有n_agent个智能体，不能按照这么来计算skill
                all_skill = [own_skill, enemy_skill, ally_skill]
                # shape of action_out_h : own: (total * skill_dim, 1, entity_embed_dim), enemy: (total * skill_dim, n_enemy, entity_embed_dim)
                action_out_h = self.mac.forward_planner_feedforward(
                    all_skill,
                    additional_input=obs_inputs_expand,
                    forward_type="action",
                    task=task
                    )

                # 2. 得到真实动作序列的表征
                gt_act_seq = act_outs.squeeze(2) if act_outs.dim() == 5 else act_outs  # [bs, c, n_agents, a]
                gt_act_seq = gt_act_seq.permute(0, 2, 1, 3).reshape(total, self.c * a)  # [total, c*a]
                # 3. 投影
                proj_skill = self.infonce_proj_skill(th.cat(action_out_h, 1).reshape(total*skill_dim, -1)).reshape(skill_dim, total, -1)  # [skill_dim, total, proj_dim]
                # 投影动作序列: [total, c*a] -> [total, proj_dim]
                proj_act = self.infonce_proj_act(gt_act_seq)

                # 4. logits: [total, skill_dim]，labels: [total]（真实skill index）
                logits = []
                for i in range(total):
                    sims = []
                    for k in range(skill_dim):
                        sim = F.cosine_similarity(proj_act[i], proj_skill[k, i], dim=0)
                        sims.append(sim)
                    logits.append(th.stack(sims))
                logits = th.stack(logits, dim=0)  # [total, skill_dim]
                labels = skill_index.reshape(-1)  # [total]
                infonce_loss += F.cross_entropy(logits, labels)
            # ====== InfoNCE部分结束 ======
            t += self.c

        if (# TODO:不将ssl_type进行设置，就可以不加入contrastive loss进行任务区分；或者说，我的ssl_loss是为了学习多个skill-state在多个t后的状态表示而非任务区分，需要修改
            len(self.last_batch) != 0   # TODO: 那如果要继续multi-task的配置，是否可以使用premier-TACO的表示学习方法?
            and self.ssl_type == "moco"
            and not self.main_args.adaptation
        ):
            ssl_loss = 0.0
            self.mac.init_hidden(batch.batch_size, task)

            cur_random = random.randint(0, self.task2n_agents[task] - 1)
            pos_random = random.randint(0, self.task2n_agents[task] - 1)
            while pos_random == cur_random:
                pos_random = random.randint(0, self.task2n_agents[task] - 1)
            cur_t = random.randint(0, batch.max_seq_length - self.c - 1)

            mac_out, _ = self.mac.forward_discriminator(batch, t=cur_t, task=task)
            cur_out, pos_out = mac_out[:, cur_random], mac_out[:, pos_random]

            total_target = []
            for i, task_ in enumerate(self.last_batch):
                if task_ == task:
                    continue
                target_outs = self.compute_neg_sample(self.last_batch[task_], task_)
                total_target.append(target_outs)
            total_target = th.cat(total_target, dim=0)

            for _ in range(cur_out.shape[0]):
                ssl_loss += self.contrastive_loss(# TODO:这里是不是写错了？区分任务应该是和total_target做contrastive loss吧. 不过在我的实现中应该不需要这个损失
                    cur_out, pos_out.detach(), target_outs.detach()
                )
            ssl_loss = ssl_loss / cur_out.shape[0]

        elif (
            len(self.last_batch) != 0
            and self.ssl_type == "byol"
            and not self.main_args.adaptation
        ):
            ssl_loss = 0.0
            cur_outs, pos_outs = [], []
            target_cur_outs, target_pos_outs = [], []
            self.mac.init_hidden(batch.batch_size, task)
            self.target_mac.init_hidden(batch.batch_size, task)

            for t in range(batch.max_seq_length - self.c):
                mac_out, _ = self.mac.forward_discriminator(batch, t=t, task=task)
                cur_random = random.randint(0, self.task2n_agents[task] - 1)
                pos_random = random.randint(0, self.task2n_agents[task] - 1)
                while pos_random == cur_random:
                    pos_random = random.randint(0, self.task2n_agents[task] - 1)
                cur_out, pos_out = mac_out[:, cur_random], mac_out[:, pos_random]
                cur_outs.append(cur_out)
                pos_outs.append(pos_out)

                with th.no_grad():
                    target_mac_out, _ = self.target_mac.forward_discriminator(
                        batch, t=t, task=task
                    )
                    cur_random = random.randint(0, self.task2n_agents[task] - 1)
                    pos_random = random.randint(0, self.task2n_agents[task] - 1)
                    while pos_random == cur_random:
                        pos_random = random.randint(0, self.task2n_agents[task] - 1)
                    target_cur_out, target_pos_out = (
                        target_mac_out[:, cur_random],
                        target_mac_out[:, pos_random],
                    )
                    target_cur_outs.append(target_cur_out)
                    target_pos_outs.append(target_pos_out)

            n_random = random.randint(0, self.ssl_tw - 1)
            cur_outs, pos_outs = th.stack(cur_outs, dim=1), th.stack(pos_outs, dim=1)
            target_cur_outs, target_pos_outs = th.stack(
                target_cur_outs, dim=1
            ), th.stack(target_pos_outs, dim=1)
            if n_random != 0 and n_random < batch.max_seq_length - self.c:
                cur_outs, pos_outs = cur_outs[:, :-n_random].reshape(
                    -1, self.entity_embed_dim
                ), pos_outs[:, n_random:].reshape(-1, self.entity_embed_dim)
                target_cur_outs, target_pos_outs = target_cur_outs[
                    :, :-n_random
                ].reshape(-1, self.entity_embed_dim), target_pos_outs[
                    :, n_random:
                ].reshape(
                    -1, self.entity_embed_dim
                )

            ssl_loss = (
                self.l2_loss(cur_outs, target_pos_outs.detach()).mean()
                + self.l2_loss(target_cur_outs.detach(), pos_outs).mean()
            ) / 2

        else:
            ssl_loss = th.tensor(0.0)
        # vae_loss实际上就是另一种形式的dec_loss. 所以只在planner中记录dec_loss就行了
        vae_loss = dec_loss / (batch.max_seq_length - self.c)
        infonce_loss = infonce_loss / (batch.max_seq_length - self.c)
        loss = vae_loss + infonce_loss
        if ssl_loss is not None:
            loss += self.beta * ssl_loss
        loss.backward()
        return vae_loss, ssl_loss, infonce_loss

    def test_vae(self, batch: EpisodeBatch, t_env: int, episode_num: int, task: str):
        rewards = batch["reward"][:, :]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :].float()
        mask = batch["filled"][:, :].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # # Calculate estimated Q-Values
        self.mac.init_hidden(batch.batch_size, task)

        dec_loss = 0.0  ### batch time agent skill
        self.mac.init_hidden(batch.batch_size, task)
        for t in range(batch.max_seq_length - self.c):
            seq_action_output = self.mac.forward_seq_action(batch, t, task=task)
            b, c, n, a = seq_action_output.size()
            dec_loss += (
                F.cross_entropy(
                    seq_action_output.reshape(-1, a),
                    actions[:, t : t + self.c].squeeze(-1).reshape(-1),
                    reduction="sum",
                )
                / mask[:, t : t + self.c].sum()
            ) / n

        vae_loss = dec_loss / (batch.max_seq_length - self.c)
        loss = vae_loss

        # 添加wandb记录
        # if self.use_wandb:
        #     wandb.log({f"{task}/test_vae_loss": loss.item()}, step=t_env)

        self.logger.log_stat(f"train/{task}/test_vae_loss", loss.item(), t_env)
    # 使用value-based的方法训练MARL的值函数
    def train_value(self, batch: EpisodeBatch, t_env: int, episode_num: int, task: str):
        self.mac.agent.value.requires_grad_(True)
        # Get the relevant quantities
        rewards = batch["reward"][:, :]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :].float()
        mask = batch["filled"][:, :].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        #### value net inference
        values = []
        target_values = []
        self.mac.init_hidden(batch.batch_size, task)
        self.target_mac.init_hidden(batch.batch_size, task)
        for t in range(batch.max_seq_length):
            value = self.mac.forward_value(batch, t=t, task=task)
            values.append(value)
            with th.no_grad():
                target_value = self.target_mac.forward_value(batch, t=t, task=task)
                target_values.append(target_value)

        # bs, t_len, n_agents, 1
        values = th.stack(values, dim=1)
        target_values = th.stack(target_values, dim=1)
        rewards = rewards.reshape(-1, batch.max_seq_length, 1)

        if self.mixer is not None:
            mixed_values = self.mixer(
                values, batch["state"][:, :], self.task2decomposer[task]
            )
            with th.no_grad():
                target_mixed_values = self.target_mixer(
                    target_values, batch["state"][:, :], self.task2decomposer[task]
                ).detach()
        else:
            mixed_values = values.sum(dim=2)
            target_mixed_values = target_values.sum(dim=2).detach()

        cs_rewards = batch["reward"].clone()
        discount = self.main_args.gamma
        for i in range(1, self.c):
            cs_rewards[:, : -self.c] += discount * rewards[:, i : -(self.c - i)]
            discount *= self.main_args.gamma

        td_error = (
            mixed_values[:, : -self.c]
            - cs_rewards[:, : -self.c]
            - discount
            * (1 - terminated[:, self.c - 1 : -1])
            * target_mixed_values[:, self.c :].detach()
        )
        mask = mask.expand_as(mixed_values)
        masked_td_error = td_error * mask[:, : -self.c]

        if self.adaptation:
            value_loss = th.mean((masked_td_error**2).sum()) / mask[:, : -self.c].sum()
        else:
            value_loss = (
                th.mean( #Implicit Q-learning objective的不对称权重:当TD误差<0（高估）时，权重更接近1-self.epsilon;当TD误差>0（低估）时，权重更接近self.epsilon
                    th.abs(self.epsilon - (masked_td_error < 0).float()).mean()
                    * (masked_td_error**2).sum()
                )
                / mask[:, : -self.c].sum()
            )

        loss = value_loss

        # 添加wandb记录
        # 都只在train_planner中记录
        # if self.use_wandb:
        #     wandb.log({f"{task}/value_loss": value_loss.item()}, step=t_env)

        loss.backward()

        return value_loss
        ####

    def train_planner(
        self,
        batch: EpisodeBatch,
        t_env: int,
        episode_num: int,
        task: str,
        v_loss=th.tensor(0.0),
        dec_loss=th.tensor(0.0),
        cls_loss=th.tensor(0.0),
        ssl_loss=th.tensor(0.0),
        infonce_loss=th.tensor(0.0),
    ):
        self.mac.agent.value.requires_grad_(False)
        # Get the relevant quantities
        rewards = batch["reward"][:, :]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :].float()
        mask = batch["filled"][:, :].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        mac_value = []
        mac_reward = []
        planner_loss = th.tensor(0.0, device=self.device)
        reward_pred_loss = th.tensor(0.0, device=self.device)  # 添加奖励预测损失
        b, t, n = actions.shape[0], actions.shape[1], actions.shape[2]
        self.mac.init_hidden(batch.batch_size, task)
        self.target_mac.init_hidden(batch.batch_size, task)
        for t in range(batch.max_seq_length - self.c):
            out_h, obs_loss, _, loss_dict = self.mac.forward_planner(
                batch,
                t=t,
                task=task,
                actions=actions[:, t],
                training=True,
                loss_out=True,
            )
            # 预测值函数
            # agent_inputs = self.mac._build_inputs(batch, t=t, task=task)
            agent_inputs = batch["state"][:, t]
            value_out_h = self.mac.forward_planner_feedforward(
                out_h, additional_input=agent_inputs, forward_type="value", task=task
            )
            mac_value.append(value_out_h)
            
            # 添加奖励预测部分
            reward_out_h = self.mac.forward_planner_feedforward(
                out_h, additional_input=agent_inputs, forward_type="reward", task=task
            )
            mac_reward.append(reward_out_h)
            # 获取实际奖励            
            planner_loss += obs_loss

        t = batch.max_seq_length - self.c
        for i in range(self.c):
            out_h, _, _ , loss_dict= self.mac.forward_planner(
                batch, t=t + i, task=task, actions=actions[:, t + i]
            )
            # agent_inputs = self.mac._build_inputs(batch, t=t + i, task=task)
            agent_inputs = batch["state"][:, t]
            value_out_h = self.mac.forward_planner_feedforward(
                out_h, additional_input=agent_inputs, forward_type="value", task=task
            )
            reward_out_h = self.mac.forward_planner_feedforward(
                out_h, additional_input=agent_inputs, forward_type="reward", task=task
            )
            mac_reward.append(reward_out_h)
            mac_value.append(value_out_h)

        #### value net inference   使用了前面的生成的skill，加上目前的state，来推断值函数，即论文里的local information
        value_pre = []           # TODO:但是它的skill对于value推断出的不是next state，我需要自己添加WM来完成这一步
        target_value_pre = []
        reward_pre = []
        for t in range(batch.max_seq_length):
            # target_value是使用skill结合计算出的（结合论文里equation 7的local information），但是当前timestep的value是没有使用skill计算的（论文里单纯用obs）。这是为什么？
            # value是用每个agent的obs计算的，然后通过mix network结合在一起，而target_value是用skill计算的，然后通过mix network结合在一起
            # 这个是最主要的value计算函数。forward_skill_value只是为了引导skill条件下的value，在最后test过程中应该是不用的
            value = self.mac.forward_value(batch, t=t, task=task)
            reward_pred = self.mac.forward_reward_skill(batch.batch_size, mac_reward[t], task=task)
            with th.no_grad():
                target_value = self.target_mac.forward_value_skill(
                    batch.batch_size, mac_value[t], task=task
                )
            value_pre.append(value)
            reward_pre.append(reward_pred)
            target_value_pre.append(target_value)

        value_pre = th.stack(value_pre, dim=1)
        # TODO:check dim
        reward_pre = th.stack(reward_pre, dim=1)
        target_value_pre = th.stack(target_value_pre, dim=1)

        if self.mixer is not None:
            mixed_values = self.mixer(
                value_pre, batch["state"][:, :], self.task2decomposer[task]
            )
            target_mixed_values = self.target_mixer(
                target_value_pre, batch["state"][:, :], self.task2decomposer[task]
            ).detach()
        else:
            mixed_values = value_pre.sum(dim=2)
            target_mixed_values = target_value_pre.sum().detach()

        cs_rewards = batch["reward"].clone()
        discount = self.main_args.gamma
        for i in range(1, self.c):
            cs_rewards[:, : -self.c] += discount * rewards[:, i : -(self.c - i)]
            discount *= self.main_args.gamma

        if self.adaptation:
            loss = -mixed_values.sum() / mask.sum()
        else:
            td_error = (
                discount * target_mixed_values[:, self.c :].detach()
                + cs_rewards[:, : -self.c]
                - mixed_values[:, : -self.c]
            )
            planner_loss = planner_loss / (batch.max_seq_length - self.c)
            mask = mask.expand_as(mixed_values)
            # TODO: 不会是这些空的位置，计算的loss也参与了梯度，所以预测值被强行绑定在了0附近？
            reward_pred_loss += F.mse_loss(
                reward_pre[:, :-self.c].reshape(-1, 1), 
                cs_rewards[:, :-self.c].reshape(-1, 1),
                reduction="sum"
            ) / mask[:, : -self.c].sum()
            td_error = (td_error * mask[:, : -self.c]).sum() / mask[:, : -self.c].sum()
            weight = th.exp(td_error * self.td_weight)
            weight = th.clamp_max(weight, 100.0).detach()
            # 论文里的equation 7，td-error与planner损失相乘
            loss = weight * planner_loss + self.main_args.reward_pred_weight * reward_pred_loss

        
        loss.backward()

        # episode_num should be pulic
        if (
            t_env - self.last_target_update_episode
        ) / self.main_args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = t_env

        if (
            t_env - self.task2train_info[task]["log_stats_t"]
            >= self.task2args[task].learner_log_interval
        ):
            self.logger.log_stat(f"{task}/dec_loss", dec_loss.item(), t_env)
            self.logger.log_stat(f"{task}/value_loss", v_loss.item(), t_env)
            self.logger.log_stat(f"{task}/plan_loss", planner_loss.item(), t_env)
            # self.logger.log_stat(f"{task}/ssl_loss", ssl_loss.item(), t_env)
            self.logger.log_stat(f"{task}/infonce_loss", infonce_loss.item(), t_env)
            self.logger.log_stat(f"{task}/reward_pred_loss", reward_pred_loss.item(), t_env)
            
            self.task2train_info[task]["log_stats_t"] = t_env

        # 在函数末尾添加wandb记录
        # TODO: 这里怎么不更新self.task2train_info[task]["log_stats_t"]?那不是每次都要记录了吗?
        if self.use_wandb:
            wandb.log({
                f"{task}/dec_loss": dec_loss.item() if dec_loss is not None else 0.0,
                f"{task}/value_loss": v_loss.item() if v_loss is not None else 0.0,
                # planner_loss就是TD-error加权过后的obs_loss
                f"{task}/plan_loss": planner_loss.item(),
                # f"{task}/ssl_loss": ssl_loss.item() if ssl_loss is not None else 0.0,
                f"{task}/infonce_loss": infonce_loss.item() if infonce_loss is not None else 0.0,
                f"{task}/reward_pred_loss": reward_pred_loss.item(),
                # f"{task}/td_error": td_error.item() if not self.adaptation else 0.0,
                # f"{task}/weight": weight.mean().item() if not self.adaptation else 0.0
            }, step=t_env)

    # 这个函数是用来训练SSL的，主要是对比损失函数，但在这个代码中没用到
    def train_ssl(self, batch: EpisodeBatch, t_env: int, episode_num: int, task: str):
        # Get the relevant quantities
        rewards = batch["reward"][:, :]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :].float()
        mask = batch["filled"][:, :].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        ssl_loss = 0.0
        target_outs, target_outs_h = [], []
        self.mac.init_hidden(batch.batch_size, task)
        self.target_mac.init_hidden(batch.batch_size, task)

        cur_random, last_random = random.randint(
            0, self.task2n_agents[task] - 1
        ), random.randint(0, self.task2n_agents[self.last_task] - 1)
        pos_random = random.randint(0, self.task2n_agents[task] - 1)
        while pos_random == cur_random:
            pos_random = random.randint(0, self.task2n_agents[task] - 1)
        cur_t = random.randint(0, batch.max_seq_length - self.c - 1)

        mac_out, _ = self.mac.forward_discriminator(batch, t=cur_t, task=task)
        cur_out, pos_out = mac_out[:, cur_random], mac_out[:, pos_random]
        for t in range(self.last_batch.max_seq_length - self.c):
            with th.no_grad():
                target_mac_out, _ = self.target_mac.forward_discriminator(
                    self.last_batch, t=t, task=self.last_task
                )
                target_mac_out = target_mac_out[:, last_random]
            target_outs.append(target_mac_out)

        target_outs = th.cat(target_outs, dim=1).reshape(
            -1, self.main_args.entity_embed_dim
        )

        for _ in range(cur_out.shape[0]):
            ssl_loss += self.contrastive_loss(
                cur_out, pos_out.detach(), target_outs.detach()
            )
        ssl_loss = ssl_loss / cur_out.shape[0]

        # 添加wandb记录
        # if self.use_wandb:
        #     wandb.log({f"{task}/ssl_loss": ssl_loss.item()}, step=t_env)

        ssl_loss.backward()

        return ssl_loss

    def pretrain(self, batch: EpisodeBatch, t_env: int, episode_num: int, task: str):
        if self.pretrain_steps == 0:
            self._reset_optimizer()
            for t in self.task2args:
                task_args = self.task2args[t]
                self.task2train_info[t]["log_stats_t"] = (
                    -task_args.learner_log_interval - 1
                )

        self.train_vae(batch, t_env, episode_num, task)
        self.pretrain_steps += 1

    def test_pretrain(
        self, batch: EpisodeBatch, t_env: int, episode_num: int, task: str
    ):
        self.test_vae(batch, t_env, episode_num, task)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, task: str, use_external_skill=False):
        if self.training_steps == 0:
            self._reset_optimizer()
            self.device = batch.device
            for t in self.task2args:
                task_args = self.task2args[t]
                self.task2train_info[t]["log_stats_t"] = (
                    -task_args.learner_log_interval - 1
                )

        if self.adaptation:
            # 替换三个独立的训练函数为一个整合的函数
            v_loss, reward_loss, rec_loss, total_loss = self.train_adaptation(batch, t_env, episode_num, task)
            self.update(pretrain=False)  # 添加这一行以应用train_adaptation中计算的梯度
            
            # self.train_planner(
            #     batch,
            #     t_env,
            #     episode_num,
            #     task,
            #     v_loss=v_loss,
            #     dec_loss=th.tensor(0.0, device=self.device),
            #     ssl_loss=th.tensor(0.0, device=self.device),
            # )
            # self.update(pretrain=False)  # 添加这一行以应用train_planner中计算的梯度
        else:
            # dec_loss = 0
            # ssl_loss = 0
            # v_loss = 0
            # TODO: 这里记得修改回来
            dec_loss, ssl_loss, infonce_loss = self.train_vae(batch, t_env, episode_num, task)
            self.update_last_batch(task, batch)
            self.update(pretrain=False)
            v_loss = self.train_value(batch, t_env, episode_num, task)
            self.update(pretrain=False)

            self.train_planner(
                batch,
                t_env,
                episode_num,
                task,
                v_loss=v_loss,
                dec_loss=dec_loss,
                ssl_loss=ssl_loss,
                infonce_loss=infonce_loss,
            )
        self.training_steps += 1

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        """将模型转移到GPU上"""
        self.infonce_proj_act.cuda()
        self.infonce_proj_skill.cuda()
        # self.device = "cuda"
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        """保存模型，mixer现在在mac中保存"""
        self.mac.save_models(path)
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        self.optimiser.load_state_dict(
            th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage)
        )

    def train_reward(self, batch: EpisodeBatch, t_env: int, episode_num: int, task: str):
        """
        训练奖励预测模型，用于在线环境适配
        此函数在adaptation模式下使用，不重新训练skill
        """
        # 获取相关数据
        rewards = batch["reward"][:, :]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :].float()
        mask = batch["filled"][:, :].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        
        # 初始化
        mac_reward = []
        reward_pred_loss = 0.0
        self.mac.init_hidden(batch.batch_size, task)
        
        # 对每个时间步
        for t in range(batch.max_seq_length - self.c):
            # 获取当前skill的embedding，但不计算梯度更新skill
            with th.no_grad():
                out_h, _, _, loss_dict = self.mac.forward_planner(
                    batch, t=t, task=task, actions=actions[:, t], training=False
                )
            
            # 使用skill预测奖励，这部分参数需要更新
            agent_inputs = batch["state"][:, t]
            reward_out_h = self.mac.forward_planner_feedforward(
                out_h, additional_input=agent_inputs, forward_type="reward", task=task
            )
            mac_reward.append(reward_out_h)
        
        # 处理最后几个时间步
        t = batch.max_seq_length - self.c
        for i in range(self.c):
            with th.no_grad():
                out_h, _, _, loss_dict = self.mac.forward_planner(
                    batch, t=t + i, task=task, actions=actions[:, t + i]
                )
            agent_inputs = batch["state"][:, t + i]
            reward_out_h = self.mac.forward_planner_feedforward(
                out_h, additional_input=agent_inputs, forward_type="reward", task=task
            )
            mac_reward.append(reward_out_h)

        # 计算奖励预测
        reward_pre = []
        for t in range(batch.max_seq_length):
            reward_pred = self.mac.forward_reward_skill(batch.batch_size, mac_reward[t], task=task)
            reward_pre.append(reward_pred)
            
        # 堆叠奖励预测
        reward_pre = th.stack(reward_pre, dim=1)
        
        # 计算累积奖励
        cs_rewards = batch["reward"].clone()
        discount = self.main_args.gamma
        for i in range(1, self.c):
            cs_rewards[:, : -self.c] += discount * rewards[:, i : -(self.c - i)]
            discount *= self.main_args.gamma
        
        # 计算奖励预测损失
        mask_expanded = mask.expand_as(reward_pre)
        reward_pred_loss = F.mse_loss(
            reward_pre[:, :-self.c].reshape(-1, 1), 
            cs_rewards[:, :-self.c].reshape(-1, 1),
            reduction="sum"
        ) / mask_expanded[:, : -self.c].sum()
        
        # 反向传播
        reward_pred_loss.backward()
        
        # 记录日志
        if (
            t_env - self.task2train_info[task]["log_stats_t"]
            >= self.task2args[task].learner_log_interval
        ):
            self.logger.log_stat(f"{task}/reward_pred_loss", reward_pred_loss.item(), t_env)
            
        # 添加wandb记录
        if self.use_wandb:
            wandb.log({
                f"{task}/reward_pred_loss": reward_pred_loss.item(),
            }, step=t_env)
            
        return reward_pred_loss

    def train_rec(self, batch: EpisodeBatch, t_env: int, episode_num: int, task: str):
        """
        训练状态转移预测模型(世界模型)，用于在线环境适配
        此函数在adaptation模式下使用，不重新训练skill
        """
        # 获取相关数据
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :].float()
        mask = batch["filled"][:, :].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        
        # 初始化
        rec_loss = 0.0
        self.mac.init_hidden(batch.batch_size, task)
        
        # 对每个时间步计算重建损失
        for t in range(batch.max_seq_length - self.c):
            # 获取当前state和下一个state
            states = batch["state"][:, t]
            next_states = batch["state"][:, t + self.c]
            
            # 获取当前obs和下一个obs
            current_obs = batch["obs"][:, t]
            next_obs = batch["obs"][:, t + self.c]
            
            # 使用已有skill进行预测，但不更新skill参数
            with th.no_grad():
                out_h, _, _, loss_dict = self.mac.forward_planner(
                    batch, t=t, task=task, actions=actions[:, t], training=False
                )
            
            # 提取需要的输入信息
            shape_info = self.task2input_shape_info[task]
            obs_dim = self.task2args[task].obs_shape
            agent_id_shape = shape_info['agent_id_shape']
            task_n_agents = self.task2n_agents[task]
            
            # 整理输入数据
            inputs = current_obs.reshape(-1, current_obs.shape[-1])
            obs_inputs = inputs[:, :obs_dim]
            agent_id_inputs = inputs[:, -agent_id_shape:]
            obs_input = th.cat([obs_inputs, agent_id_inputs], dim=-1)
            
            # 获取skill对应的表示
            n_enemy = self.task2decomposer[task].n_enemies
            n_agents = self.task2decomposer[task].n_agents
            n_ally = n_agents - 1
            
            skill_code = self.mac.agent.planner.task2unsqueeze_mlp[task](out_h[0]).reshape(
                batch.batch_size * n_agents, self.mac.get_total_agents(task), self.main_args.entity_embed_dim
            )
            own_skill = skill_code[:, 0].unsqueeze(1)
            enemy_skill = skill_code[:, 1:1+n_enemy]
            ally_skill = skill_code[:, 1+n_enemy:1+n_enemy+n_ally]
            all_skill = [own_skill, enemy_skill, ally_skill]
            
            # 计算重建损失 - 使用rec_module的pred_next直接计算下一状态
            loss = self.mac.agent.planner.rec_module(
                all_skill, obs_input, next_obs, states, next_states, task, t=t, actions=actions[:, t]
            )
            
            rec_loss += loss
        
        # 平均重建损失
        rec_loss = rec_loss / (batch.max_seq_length - self.c)
        
        # 反向传播
        rec_loss.backward()
        
        # 记录日志
        if (
            t_env - self.task2train_info[task]["log_stats_t"]
            >= self.task2args[task].learner_log_interval
        ):
            self.logger.log_stat(f"{task}/rec_loss", rec_loss.item(), t_env)
            
        # 添加wandb记录
        if self.use_wandb:
            wandb.log({
                f"{task}/rec_loss": rec_loss.item(),
            }, step=t_env)
            
        return rec_loss
    
    def train_adaptation(self, batch: EpisodeBatch, t_env: int, episode_num: int, task: str):
        """
        整合了值函数、奖励和状态转移预测的自适应训练函数
        在adaptation模式下使用，不重新训练skill
        一次性获取skill表示，同时更新三个模块
        """
        self.mac.agent.value.requires_grad_(True)
        # 获取相关数据
        rewards = batch["reward"][:, :]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :].float()
        mask = batch["filled"][:, :].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        
        # 初始化
        value_loss = th.tensor(0.0, device=self.device)
        reward_pred_loss = th.tensor(0.0, device=self.device)
        rec_loss = th.tensor(0.0, device=self.device)
        normal_value_loss = th.tensor(0.0, device=self.device)  # 新增：普通值函数损失
        
        # 第1阶段：获取技能表示并进行奖励、值函数和世界模型预测
        mac_reward = []
        mac_value = []  # 用于存储基于skill的值函数预测结果
        
        self.mac.init_hidden(batch.batch_size, task)
        self.target_mac.init_hidden(batch.batch_size, task)
        
        # 解除注释：重新启用普通值函数计算，但以self.c为步长
        values = []
        target_values = []
        for t in range(0, batch.max_seq_length, self.c):
            value = self.mac.forward_value(batch, t=t, task=task)
            values.append(value)
            with th.no_grad():
                target_value = self.target_mac.forward_value(batch, t=t, task=task)
                target_values.append(target_value)

        # 重新初始化hidden，因为forward_value_skill和forward_value是共用的，如果不重新初始化会出错
        self.mac.init_hidden(batch.batch_size, task)
        self.target_mac.init_hidden(batch.batch_size, task)
    
        # 第2阶段：获取技能表示并进行奖励和世界模型预测，同时添加值函数预测
        for t in range(0, batch.max_seq_length - self.c, self.c):
            # 获取当前state和下一个state（用于世界模型）
            states = batch["state"][:, t]
            next_states = batch["state"][:, t + self.c]
            
            # 获取当前obs和下一个obs（用于世界模型）
            current_obs = batch["obs"][:, t]
            next_obs = batch["obs"][:, t + self.c]
            
            # 从batch中获取对应时间步的skill索引
            skill_index = batch["skills"][:, t].long()  # 假设skills以 [bs, t] 的形式存储
            bs = batch.batch_size
            skill_index = skill_index.reshape(bs, -1)
            # 使用skill_index获取skill embedding
            with th.no_grad():
                # 获取skill embedding
                skill_embedding = self.mac.get_skill(skill_index)
                
                # 使用unsqueezemlp将skill embedding转换为out_h
                n_agents = self.task2n_agents[task]
                n_enemy = self.task2decomposer[task].n_enemies
                n_ally = n_agents - 1
                
                # 使用planner的unsqueeze_mlp将skill转换为实体embedding
                skill_code = self.mac.agent.planner.task2unsqueeze_mlp[task](skill_embedding)
                skill_code = skill_code.reshape(bs * n_agents, self.mac.get_total_agents(task), self.main_args.entity_embed_dim)
                
                # 拆分为own, enemy和ally的embedding
                own_skill = skill_code[:, 0].unsqueeze(1)
                enemy_skill = skill_code[:, 1:1+n_enemy]
                ally_skill = skill_code[:, 1+n_enemy:1+n_enemy+n_ally]
                out_h = [own_skill, enemy_skill, ally_skill]
        
            # 1. 使用skill预测奖励
            agent_inputs = batch["state"][:, t]
            reward_out_h = self.mac.forward_planner_feedforward(
                out_h, additional_input=agent_inputs, forward_type="reward", task=task
            )
            mac_reward.append(reward_out_h)
            
            # 2. 使用skill预测值函数
            value_out_h = self.mac.forward_planner_feedforward(
                out_h, additional_input=agent_inputs, forward_type="value", task=task
            )
            mac_value.append(value_out_h)
            
            # 3. 使用skill预测下一个状态（世界模型）
            shape_info = self.task2input_shape_info[task]
            obs_dim = self.task2args[task].obs_shape
            agent_id_shape = shape_info['agent_id_shape']
            task_n_agents = self.task2n_agents[task]
            
            # 整理输入数据
            inputs = current_obs.reshape(-1, current_obs.shape[-1])
            obs_inputs = inputs[:, :obs_dim]
            agent_id_inputs = inputs[:, -agent_id_shape:]
            obs_input = th.cat([obs_inputs, agent_id_inputs], dim=-1)
            
            # 计算重建损失
            curr_rec_loss = self.mac.agent.planner.rec_module(
                out_h, obs_input, next_obs, states, next_states, task, t=t, actions=actions[:, t]
            )
            rec_loss += curr_rec_loss
    
        # 处理最后几个时间步
        remaining_steps = batch.max_seq_length - t - self.c
        if remaining_steps > 0 and t + self.c < batch.max_seq_length:
            t = t + self.c
            with th.no_grad():
                bs = batch.batch_size
                # 获取最后一个完整时间窗口的skill索引
                skill_index = batch["skills"][:, t].long()
                skill_index = skill_index.reshape(bs, -1)
                # 获取skill embedding
                skill_embedding = self.mac.get_skill(skill_index)
                
                # 使用unsqueezemlp将skill embedding转换为out_h
                n_agents = self.task2n_agents[task]
                n_enemy = self.task2decomposer[task].n_enemies
                n_ally = n_agents - 1
                
                skill_code = self.mac.agent.planner.task2unsqueeze_mlp[task](skill_embedding)
                skill_code = skill_code.reshape(bs * n_agents, self.mac.get_total_agents(task), self.main_args.entity_embed_dim)
                
                own_skill = skill_code[:, 0].unsqueeze(1)
                enemy_skill = skill_code[:, 1:1+n_enemy]
                ally_skill = skill_code[:, 1+n_enemy:1+n_enemy+n_ally]
                out_h = [own_skill, enemy_skill, ally_skill]
        
            agent_inputs = batch["state"][:, t]
            
            # 奖励预测
            reward_out_h = self.mac.forward_planner_feedforward(
                out_h, additional_input=agent_inputs, forward_type="reward", task=task
            )
            mac_reward.append(reward_out_h)
        
            # 值函数预测
            value_out_h = self.mac.forward_planner_feedforward(
                out_h, additional_input=agent_inputs, forward_type="value", task=task
            )
            mac_value.append(value_out_h)
    
        # ----- 开始计算各个损失 -----
        
        # 1. 计算值函数损失 - 使用基于skill的估计
        # 预测当前值和目标值
        value_pre = []
        target_value_pre = []
        
        for t in range(len(mac_value)):
            # 使用当前网络和基于skill的值估计
            value = self.mac.forward_value_skill(batch.batch_size, mac_value[t], task=task)
            value_pre.append(value)
            
            # 使用目标网络和基于skill的值估计
            with th.no_grad():
                target_value = self.target_mac.forward_value_skill(batch.batch_size, mac_value[t], task=task)
                target_value_pre.append(target_value)
    
        # 堆叠值估计
        if value_pre:  # 确保有值估计再处理
            value_pre = th.stack(value_pre, dim=1)  # [bs, n_steps, n_agents, 1]
            target_value_pre = th.stack(target_value_pre, dim=1)  # [bs, n_steps, n_agents, 1]
            
            # 堆叠普通值函数估计
            stacked_values = th.stack(values, dim=1)  # [bs, n_steps, n_agents, 1]
            stacked_target_values = th.stack(target_values, dim=1)  # [bs, n_steps, n_agents, 1]
            
            # 创建按c_step为步长的索引列表
            value_pred_indices = list(range(0, batch.max_seq_length, self.c))
            
            # 如果最后一组索引已经超出范围，就移除它
            if value_pred_indices[-1] >= batch.max_seq_length:
                value_pred_indices.pop()
        
            # 获取这些时间点的状态，用于mixer
            state_inputs_for_mixer = batch["state"][:, value_pred_indices]
        
            # 使用mixer聚合多智能体值 - 基于skill的值函数
            if self.mixer is not None:
                mixed_values = self.mixer(value_pre, state_inputs_for_mixer, self.task2decomposer[task])
                with th.no_grad():
                    target_mixed_values = self.target_mixer(target_value_pre, state_inputs_for_mixer, self.task2decomposer[task]).detach()
                
                # 使用mixer聚合多智能体值 - 普通值函数
                normal_mixed_values = self.mixer(stacked_values, state_inputs_for_mixer, self.task2decomposer[task])
                with th.no_grad():
                    normal_target_mixed_values = self.target_mixer(stacked_target_values, state_inputs_for_mixer, self.task2decomposer[task]).detach()
            else:
                mixed_values = value_pre.sum(dim=2)
                with th.no_grad():
                    target_mixed_values = target_value_pre.sum(dim=2).detach()
                
                # 普通值函数
                normal_mixed_values = stacked_values.sum(dim=2)
                with th.no_grad():
                    normal_target_mixed_values = stacked_target_values.sum(dim=2).detach()
        
            # 计算累积奖励
            cs_rewards = batch["reward"].clone()
            discount = self.main_args.gamma
            for i in range(1, self.c):
                if i < batch.max_seq_length:  # 确保不会索引越界
                    cs_rewards[:, :-i] += discount * rewards[:, i:, :]
                discount *= self.main_args.gamma
                
            # 获取这些时间点的累积奖励、终止状态和掩码
            selected_rewards = cs_rewards[:, value_pred_indices]  # [bs, n_steps]
            
            # 修复：重写终止状态检查逻辑，考虑中间的终止状态
            term_mask = []
            for t in value_pred_indices:
                # 检查从t到t+self.c-1之间的所有时间步是否有终止
                if t + self.c - 1 < batch.max_seq_length:
                    # 创建初始掩码 - 假设没有终止
                    t_mask = th.ones_like(terminated[:, 0])
                    
                    # 检查区间内的每个时间步
                    for j in range(self.c):
                        if t + j < batch.max_seq_length:
                            # 如果任何时间步终止，则乘以(1-terminated)会使掩码为0
                            t_mask = t_mask * (1 - terminated[:, t + j])
                
                    term_mask.append(t_mask)
                else:
                    # 如果超出序列长度，视为终止
                    term_mask.append(th.zeros_like(terminated[:, 0]))
        
            term_mask = th.stack(term_mask, dim=1)  # [bs, n_steps]
        
            # 获取这些时间点的掩码
            step_mask = mask[:, value_pred_indices]  # [bs, n_steps]
        
            # 调整维度以匹配值函数输出
            if mixed_values.dim() > step_mask.dim():
                selected_rewards = selected_rewards.unsqueeze(-1)  # [bs, n_steps, 1]
                term_mask = term_mask.unsqueeze(-1)  # [bs, n_steps, 1]
                step_mask = step_mask.unsqueeze(-1)  # [bs, n_steps, 1]
        
            # 计算TD目标和TD误差 - 基于skill的值函数
            if mixed_values.shape[1] > 1:
                # TD目标：r_t + gamma^c * V(s_{t+c}) * (1-done_{t+c-1})
                td_targets = selected_rewards[:, :-1] + self.main_args.gamma**self.c * term_mask[:, :-1] * target_mixed_values[:, 1:].detach()
                
                # TD误差：V(s_t) - TD目标
                td_error = mixed_values[:, :-1] - td_targets
                
                # 应用掩码
                masked_td_error = td_error * step_mask[:, :-1]
                
                # 计算均方误差损失
                value_loss = th.sum(masked_td_error**2) / (step_mask[:, :-1].sum() + 1e-9)
                
                # 新增：计算普通值函数的TD损失
                normal_td_error = normal_mixed_values[:, :-1] - td_targets  # 使用相同的TD目标
                masked_normal_td_error = normal_td_error * step_mask[:, :-1]
                normal_value_loss = th.sum(masked_normal_td_error**2) / (step_mask[:, :-1].sum() + 1e-9)
                
                # 新增：教师-学生损失，让基于技能的值函数指导普通值函数
                guidance_error = normal_mixed_values[:, :-1] - mixed_values[:, :-1].detach()
                masked_guidance_error = guidance_error * step_mask[:, :-1]
                guidance_loss = th.sum(masked_guidance_error**2) / (step_mask[:, :-1].sum() + 1e-9)
                
                # 合并普通值函数的两种损失
                normal_value_loss = 0.5 * normal_value_loss + 0.5 * guidance_loss
            else:
                # 处理只有一个时间步的情况
                value_loss = th.tensor(0.0, device=self.device)
                normal_value_loss = th.tensor(0.0, device=self.device)
        
        # 2. 计算奖励预测损失
        reward_pre = []
        for reward_out_h in mac_reward:
            reward_pred = self.mac.forward_reward_skill(batch.batch_size, reward_out_h, task=task)
            reward_pre.append(reward_pred)
        
        if reward_pre:
            reward_pre = th.stack(reward_pre, dim=1)
        
            # 创建reward_mask以匹配reward_pre的形状
            reward_mask = th.zeros_like(reward_pre)

            # 填充mask，只在有效的预测点上为1
            for i, t in enumerate(range(0, batch.max_seq_length, self.c)):
                if i < reward_pre.shape[1]:  # 确保不超出reward_pre的范围
                    reward_mask[:, i] = mask[:, t]
            
            # 创建按c_step为步长的索引列表
            reward_pred_indices = list(range(0, batch.max_seq_length, self.c))
            
            # 如果最后一组索引已经超出范围，就移除它
            if reward_pred_indices[-1] >= batch.max_seq_length:
                reward_pred_indices.pop()
            
            # 计算累积奖励 - 仅在执行奖励预测的时间点
            cs_rewards = batch["reward"].clone()
            discount = self.main_args.gamma
            for i in range(1, self.c):
                if i < batch.max_seq_length:  # 确保不会索引越界
                    cs_rewards[:, :-i] += discount * rewards[:, i:, :]
                discount *= self.main_args.gamma
                
            # 提取选定时间步的预测和真实奖励
            selected_preds = reward_pre
            selected_targets = cs_rewards[:, reward_pred_indices]
            selected_masks = mask[:, reward_pred_indices]

            # 计算奖励预测损失
            reward_pred_loss = F.mse_loss(
                selected_preds.reshape(-1, 1),
                selected_targets.reshape(-1, 1),
                reduction="none"
            ) * selected_masks.reshape(-1, 1)
            reward_pred_loss = reward_pred_loss.sum() / (selected_masks.sum() + 1e-8)
        
    # 3. 计算状态转移预测损失（已经在循环中累计)
        rec_loss = rec_loss / max(1, (batch.max_seq_length - self.c)//self.c)
        
        # 计算总损失，加入普通值函数损失
        total_loss = value_loss + reward_pred_loss + rec_loss + normal_value_loss
        
        # 反向传播
        total_loss.backward()
        
        # 记录日志
        if (
            t_env - self.task2train_info[task]["log_stats_t"]
            >= self.task2args[task].learner_log_interval
        ):
            self.logger.log_stat(f"{task}/value_loss", value_loss.item(), t_env)
            self.logger.log_stat(f"{task}/normal_value_loss", normal_value_loss.item(), t_env)
            self.logger.log_stat(f"{task}/reward_pred_loss", reward_pred_loss.item(), t_env)
            self.logger.log_stat(f"{task}/rec_loss", rec_loss.item(), t_env)
            self.logger.log_stat(f"{task}/total_adaptation_loss", total_loss.item(), t_env)
            
        # 添加wandb记录
        if self.use_wandb:
            wandb.log({
                f"{task}/value_loss": value_loss.item(),
                f"{task}/normal_value_loss": normal_value_loss.item(),  # 添加普通值函数损失记录
                f"{task}/reward_pred_loss": reward_pred_loss.item(),
                f"{task}/rec_loss": rec_loss.item(),
                f"{task}/total_adaptation_loss": total_loss.item(),
            }, step=t_env)
        
        if (
            episode_num - self.last_target_update_episode
        ) / self.main_args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num
            
        return value_loss, reward_pred_loss, rec_loss, total_loss
