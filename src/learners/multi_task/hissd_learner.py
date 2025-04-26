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


class HISSDLearner:
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
        # 使用mac中的mixer而不是在这里初始化
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
        self.infonce_proj_skill = th.nn.Sequential(
            th.nn.Linear(self.entity_embed_dim + input_dim, proj_dim),
            th.nn.ReLU(inplace=True),
            th.nn.Linear(proj_dim, proj_dim)
        )
        # TODO: 因为只有一个task，所以这里直接用task是可以的。不过multi-task的实现中，这样是不可以的
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
        dec_loss = 0.0
        infonce_loss = 0.0
        b, t, n = actions.shape[0], actions.shape[1], actions.shape[2]
        self.mac.init_hidden(batch.batch_size, task)
        t = 0
        while t < batch.max_seq_length - self.c:
            act_outs = []
            agent_inputs = self.mac._build_inputs(batch, t=t, task=task)
            # 得到当前batch的skill embedding和skill index
            agent_outs, _, skill_index = self.mac.forward_planner(
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
            skill_dim = self.skill_dim
            bs = batch.batch_size
            n_agents = self.task2n_agents[task]
            device = agent_inputs.device
            total = bs * n_agents
            n_enemy = self.task2decomposer[task].n_enemies
            n_ally = n_agents - 1

            # 1. 构造所有skill embedding
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
            skill_obs_cat = th.cat([all_skill_embeds, obs_inputs_expand], dim=-1)  # [skill_dim, total, entity_embed_dim+obs_dim]
            # ---------------------------------------------------------------------------
            all_skill_embeds = th.stack(all_skill_embeds, dim=0)  # [skill_dim, total, entity_embed_dim]

            # 获取当前obs（agent_inputs），shape [total, obs_dim]
            obs_dim = self.task2input_shape_info[task]["input_shape"]
            obs_inputs = agent_inputs  # [total, obs_dim]

            # 拼接obs到skill embedding
            obs_inputs_expand = obs_inputs.unsqueeze(0).expand(skill_dim, -1, -1)  # [skill_dim, total, obs_dim]
            skill_obs_cat = th.cat([all_skill_embeds, obs_inputs_expand], dim=-1)  # [skill_dim, total, entity_embed_dim+obs_dim]

            # 3. 投影
            proj_skill = self.infonce_proj_skill(skill_obs_cat)

            # ---------------------------------------------------------------------------

            # 3. 投影
            proj_skill = self.infonce_proj_skill(skill_obs_cat)
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

        self.mac.agent.value.requires_grad_(True)
        loss.backward()

        return value_loss
        ####

    def train_planner(
        self,
        batch: EpisodeBatch,
        t_env: int,
        episode_num: int,
        task: str,
        v_loss=None,
        dec_loss=None,
        cls_loss=None,
        ssl_loss=None,
        infonce_loss=None,
    ):
        # Get the relevant quantities
        rewards = batch["reward"][:, :]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :].float()
        mask = batch["filled"][:, :].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        mac_value = []
        mac_reward = []
        planner_loss = 0.0
        reward_pred_loss = 0.0  # 添加奖励预测损失
        b, t, n = actions.shape[0], actions.shape[1], actions.shape[2]
        self.mac.init_hidden(batch.batch_size, task)
        self.target_mac.init_hidden(batch.batch_size, task)
        for t in range(batch.max_seq_length - self.c):
            out_h, obs_loss, _ = self.mac.forward_planner(
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
            out_h, _, _ = self.mac.forward_planner(
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

        self.mac.agent.value.requires_grad_(False)
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

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, task: str):
        if self.training_steps == 0:
            self._reset_optimizer()
            self.device = batch.device
            for t in self.task2args:
                task_args = self.task2args[t]
                self.task2train_info[t]["log_stats_t"] = (
                    -task_args.learner_log_interval - 1
                )

        if self.adaptation:
            v_loss = self.train_value(batch, t_env, episode_num, task)
            self.train_planner(
                batch,
                t_env,
                episode_num,
                task,
                v_loss=th.tensor(0.0),
                dec_loss=th.tensor(0.0),
                ssl_loss=th.tensor(0.0),
            )
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
