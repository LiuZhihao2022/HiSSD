import copy
import torch as th
import torch.nn.functional as F
from torch.optim import RMSprop, Adam, AdamW
import numpy as np
# import wandb # Add if wandb logging is explicitly enabled and used

from components.episode_buffer import EpisodeBatch

class HISSDLearner:
    def __init__(self, mac, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.use_wandb = getattr(args, "use_wandb", False)

        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.entity_embed_dim = args.entity_embed_dim
        self.skill_dim = args.skill_dim
        self.c_step = args.c_step
        
        self.mixer = mac.mixer
        self.target_mixer = mac.target_mixer

        self.params = list(mac.parameters())
        if self.mixer is not None:
            self.params += list(self.mixer.parameters())

        self.last_target_update_episode = 0
        self._reset_optimizer()

        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -args.learner_log_interval - 1

        # Hyperparameters
        self.beta = getattr(args, "beta", 1.0) # For potential SSL if added later
        self.alpha = getattr(args, "coef_conservative", 0.5)
        self.phi = getattr(args, "coef_dist", 0.1)
        self.kl_weight = getattr(args, "coef_kl", 0.01)
        
        self.td_weight = getattr(args, "td_weight", 1.0)
        self.adaptation = getattr(args, "adaptation", False)
        self.epsilon = getattr(args, "epsilon", 0.1) # For IQL-style loss

        self.pretrain_steps = 0
        self.training_steps = 0
        self.device = self.mac.device

        # InfoNCE projection MLPs
        proj_dim = getattr(args, "infonce_proj_dim", 128)
        self.infonce_coef = getattr(args, "infonce_coef", 0.1)
        self.reward_loss_coef = getattr(args, "reward_loss_coef", 0.1)
        
        self.infonce_proj_skill = th.nn.Sequential(
            th.nn.Linear(self.entity_embed_dim, proj_dim),
            th.nn.ReLU(inplace=True),
            th.nn.Linear(proj_dim, proj_dim)
        ).to(self.device)
        self.params += list(self.infonce_proj_skill.parameters())

        self.infonce_proj_act = th.nn.Sequential(
            th.nn.Linear(self.c_step * self.n_actions, proj_dim),
            th.nn.ReLU(inplace=True),
            th.nn.Linear(proj_dim, proj_dim)
        ).to(self.device)
        self.params += list(self.infonce_proj_act.parameters())
        
        # Re-initialize optimizer with all parameters
        self._reset_optimizer()


    def _reset_optimizer(self):
        if self.args.optim_type.lower() == "rmsprop":
            self.pre_optimiser = RMSprop(
                params=self.params, lr=self.args.lr, alpha=self.args.optim_alpha,
                eps=self.args.optim_eps, weight_decay=self.args.weight_decay
            )
            self.optimiser = RMSprop(
                params=self.params, lr=self.args.critic_lr if hasattr(self.args, 'critic_lr') else self.args.lr, # Use critic_lr if available
                alpha=self.args.optim_alpha, eps=self.args.optim_eps, weight_decay=self.args.weight_decay
            )
        elif self.args.optim_type.lower() == "adam":
            self.pre_optimiser = Adam(
                params=self.params, lr=self.args.lr, weight_decay=self.args.weight_decay
            )
            self.optimiser = Adam(
                params=self.params, lr=self.args.critic_lr if hasattr(self.args, 'critic_lr') else self.args.lr, 
                weight_decay=self.args.weight_decay
            )
        elif self.args.optim_type.lower() == "adamw":
            self.pre_optimiser = AdamW(
                params=self.params, lr=self.args.lr, weight_decay=self.args.weight_decay
            )
            self.optimiser = AdamW(
                params=self.params, lr=self.args.critic_lr if hasattr(self.args, 'critic_lr') else self.args.lr, 
                weight_decay=self.args.weight_decay
            )
        else:
            raise ValueError(f"Invalid optimiser type: {self.args.optim_type}")
        
        if hasattr(self, 'pre_optimiser'): # Ensure they are created before zero_grad
            self.pre_optimiser.zero_grad()
            self.optimiser.zero_grad()

    def zero_grad(self):
        if hasattr(self, 'pre_optimiser'):
            self.pre_optimiser.zero_grad()
            self.optimiser.zero_grad()

    def update(self, pretrain=True):
        grad_norm_clip = getattr(self.args, "grad_norm_clip", 10.0)
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, grad_norm_clip)
        if pretrain:
            self.pre_optimiser.step()
            self.pre_optimiser.zero_grad()
        else:
            self.optimiser.step()
            self.optimiser.zero_grad()
        return grad_norm.item() if isinstance(grad_norm, th.Tensor) else grad_norm


    def train_vae(self, batch: EpisodeBatch, t_env: int, episode_num: int, use_external_skill=False):
        rewards = batch["reward"]
        actions = batch["actions_onehot"] # Assuming actions are one-hot for projection
        terminated = batch["terminated"].float()
        mask = batch["filled"].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        
        b, max_t, _, _ = actions.shape # batch_size, max_seq_length, n_agents, n_actions
        
        self.mac.init_hidden(b)
        
        total_planner_model_loss = th.tensor(0.0, device=self.device)
        total_action_rec_loss = th.tensor(0.0, device=self.device)
        total_infonce_loss = th.tensor(0.0, device=self.device)
        steps_counted = 0

        for t_start in range(0, max_t - self.c_step, self.c_step): # Iterate in c_step chunks
            # Planner model loss (reconstruction, VQ)
            # skill_input_actions are raw action *indices* if use_external_skill is True
            skill_input_actions = batch["actions"][:, t_start] if use_external_skill else None

            # out_h is skill_embedding, planner_loss is combined (rec+commit+diver), skill_idx is from VQ
            skill_embedding, planner_loss_step, skill_idx_from_planner, loss_dict_planner = self.mac.forward_planner(
                batch, t_start, 
                actions=batch["actions"][:, t_start], # Current actions for planner context
                training=True, loss_out=True, skill_index_out=True,
                external_skill_index=skill_input_actions
            )
            current_mask_sum = mask[:, t_start : t_start + self.c_step].sum()
            if current_mask_sum > 0:
                total_planner_model_loss += planner_loss_step * current_mask_sum


            # Action reconstruction loss
            action_rec_loss_step = th.tensor(0.0, device=self.device)
            if skill_idx_from_planner is not None: # Only if VQ skill is used and index is available
                for i in range(self.c_step):
                    if t_start + i < max_t:
                        # mac.forward_action_skill expects skill_index (not one-hot)
                        # It internally gets the skill embedding, then calls agent.forward
                        # HISSDController.forward_action_skill returns agent_outs (logits)
                        pred_actions_logits = self.mac.forward_action_skill(
                            batch, t_start + i, skill_idx_from_planner.detach() # Detach index if not optimizing VQ through this path
                        ) # Shape: (bs, n_agents, n_actions)
                        
                        true_actions_idx = batch["actions"][:, t_start + i].long() # Shape: (bs, n_agents)
                        
                        # Reshape for cross-entropy: (bs * n_agents, n_actions) and (bs * n_agents)
                        pred_actions_logits_flat = pred_actions_logits.reshape(-1, self.n_actions)
                        true_actions_idx_flat = true_actions_idx.reshape(-1)
                        
                        step_mask = mask[:, t_start + i].unsqueeze(-1).expand_as(batch["actions"][:, t_start + i]).reshape(-1) # bs * n_agents
                        
                        ce_loss = F.cross_entropy(pred_actions_logits_flat, true_actions_idx_flat, reduction='none')
                        masked_ce_loss = (ce_loss * step_mask).sum() / (step_mask.sum() + 1e-8)
                        action_rec_loss_step += masked_ce_loss
                
                if self.c_step > 0 and current_mask_sum > 0:
                     total_action_rec_loss += (action_rec_loss_step / self.c_step) * current_mask_sum


            # InfoNCE loss (skill embedding vs. action sequence)
            if skill_embedding is not None and skill_embedding.shape[0] > 0:
                # actions is batch["actions_onehot"] (b, max_t, n_agents, n_actions_per_agent)
                # self.n_actions is n_actions_per_agent
                # We use the first agent's actions for the sequence projection.
                if actions.shape[2] > 0: # Ensure n_agents > 0
                    action_seq_for_proj = actions[:, t_start : t_start + self.c_step, 0, :].reshape(b, self.c_step * self.n_actions)
                else: # Fallback for n_agents = 0 or unexpected shape, create zeros
                    action_seq_for_proj = th.zeros(b, self.c_step * self.n_actions, device=actions.device, dtype=actions.dtype)


                proj_skill = self.infonce_proj_skill(skill_embedding) # (b, proj_dim)
                proj_act = self.infonce_proj_act(action_seq_for_proj)    # (b, proj_dim)

                logits = th.matmul(proj_skill, proj_act.T) # (b, b)
                infonce_loss_step = F.cross_entropy(logits, th.arange(b, device=self.device))
                if current_mask_sum > 0:
                    total_infonce_loss += infonce_loss_step * current_mask_sum # Weighted by mask sum for consistency
            
            if current_mask_sum > 0:
                steps_counted += current_mask_sum


        if steps_counted == 0:
            steps_counted = 1 # Avoid division by zero

        avg_planner_model_loss = total_planner_model_loss / steps_counted
        avg_action_rec_loss = total_action_rec_loss / steps_counted
        avg_infonce_loss = total_infonce_loss / steps_counted
        
        loss = avg_planner_model_loss + avg_action_rec_loss + self.infonce_coef * avg_infonce_loss
        
        if (~th.isnan(loss)).all() and (~th.isinf(loss)).all():
            loss.backward()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("pretrain/planner_model_loss", avg_planner_model_loss.item(), t_env)
            self.logger.log_stat("pretrain/action_rec_loss", avg_action_rec_loss.item(), t_env)
            self.logger.log_stat("pretrain/infonce_loss", avg_infonce_loss.item(), t_env)
            self.logger.log_stat("pretrain/total_loss", loss.item(), t_env)
            if loss_dict_planner is not None: # Log components from the last step
                 for k, v in loss_dict_planner.items():
                    self.logger.log_stat(f"pretrain/planner_{k}", v, t_env)

        return avg_planner_model_loss, avg_action_rec_loss, avg_infonce_loss


    def test_vae(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        self.mac.init_hidden(batch.batch_size)
        # Assuming actions are not needed for test_vae, or using actual actions if available
        # For simplicity, not passing external_skill_index for test
        _, planner_loss, _, loss_dict_planner = self.mac.forward_planner(
            batch, t=0, training=False, loss_out=True, skill_index_out=False 
        )
        self.logger.log_stat("test/vae_planner_loss", planner_loss.item(), t_env)
        if loss_dict_planner is not None:
            for k, v in loss_dict_planner.items():
                self.logger.log_stat(f"test/planner_{k}", v, t_env)
        return planner_loss


    def train_value(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        rewards = batch["reward"]
        actions = batch["actions"] # Not used directly here, but for context
        terminated = batch["terminated"].float()
        mask = batch["filled"].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        # self.mac.agent.value.requires_grad_(True) # Manage grads via optimizer for self.params

        values = []
        target_values = []
        self.mac.init_hidden(batch.batch_size)
        self.target_mac.init_hidden(batch.batch_size)

        for t_step in range(batch.max_seq_length):
            value_outs = self.mac.forward_value(batch, t_step) # (bs, n_agents, 1)
            values.append(value_outs)
            with th.no_grad():
                target_value_outs = self.target_mac.forward_value(batch, t_step)
                target_values.append(target_value_outs)
        
        values = th.stack(values, dim=1) # (bs, T, n_agents, 1)
        target_values = th.stack(target_values, dim=1) # (bs, T, n_agents, 1)

        if self.mixer is not None:
            mixed_values = self.mixer(values, batch["state"])
            with th.no_grad():
                target_mixed_values = self.target_mixer(target_values, batch["state"])
        else: # Assuming value is already global or sum over agents
            mixed_values = values.sum(dim=2) 
            target_mixed_values = target_values.sum(dim=2).detach()

        # N-step returns (using c_step as N)
        cs_rewards = rewards.clone() # (bs, T, 1)
        discount_factor = self.args.gamma
        
        # Calculate N-step rewards (sum of discounted rewards over c_step)
        # td_lambda returns: G_t = r_t + gamma*r_{t+1} + ... + gamma^{N-1}r_{t+N-1} + gamma^N * V(s_{t+N})
        # Here, we are calculating returns for Q(s_t, a_t) using Q(s_{t+N}, a_{t+N})
        # The td_error will be: Q(s_t) - (sum_{i=0}^{N-1} gamma^i r_{t+i} + gamma^N Q_target(s_{t+N}))
        
        # Let's use standard 1-step TD for value learning for simplicity first, can extend to N-step later.
        # Q_t - (r_t + gamma * (1-terminated_{t+1}) * Q_target_{t+1})
        # For N-step (like in ma_gumbel_learner for value):
        # td_error = mixed_values[:, :-self.c_step] - (cs_rewards_summed + discount^N * (1-term) * target_mixed_values[:, self.c_step:])
        
        # N-step TD error
        q_taken = mixed_values[:, : -self.c_step] # (bs, T-c_step, 1)

        # Calculate N-step rewards: sum_{k=0}^{N-1} gamma^k * r_{t+k}
        n_step_rewards_sum = th.zeros_like(rewards[:, : -self.c_step]) # (bs, T-c-step, 1)
        current_gamma = 1.0
        for k_step in range(self.c_step):
            n_step_rewards_sum += current_gamma * rewards[:, k_step : batch.max_seq_length - self.c_step + k_step]
            current_gamma *= self.args.gamma
        
        q_target_next_N_step = target_mixed_values[:, self.c_step:].detach() # (bs, T-c-step, 1)

        # 为每个时间步创建termination mask
        seq_length = batch.max_seq_length
        term_masks_list = []
        for t in range(seq_length - self.c_step):
            # 检查从t到t+self.c_step-1之间的所有时间步是否有终止
            term_mask_t = th.ones_like(terminated[:, 0, 0]) # Shape: (bs)
            for j in range(self.c_step):
                # terminated is (bs, T, 1). So terminated[:, t + j, 0] is (bs)
                if t + j < seq_length: # Ensure we don't go out of bounds for terminated
                    term_mask_t = term_mask_t * (1 - terminated[:, t + j, 0])
            term_masks_list.append(term_mask_t)
        
        term_masks = th.stack(term_masks_list, dim=1)  # [bs, seq_length-self.c_step]
        term_masks = term_masks.unsqueeze(-1) # [bs, seq_length-self.c_step, 1]
        
        td_targets = n_step_rewards_sum + \
                     (self.args.gamma**self.c_step) * \
                     term_masks * \
                     q_target_next_N_step
        
        td_error = q_taken - td_targets.detach() # Detach targets
        
        masked_td_error = td_error * mask[:, : -self.c_step]
        # TODO: only adaption mode now

        # if self.adaptation:
        #      value_loss = (masked_td_error**2).sum() / (mask[:, : -self.c_step].sum() + 1e-8)
        # else: # IQL-style loss
        #     # abs_td_error = th.abs(masked_td_error) # Not directly used in this IQL form
        #     loss_weight = th.abs(self.epsilon - (masked_td_error < 0).float())
        #     value_loss = (loss_weight * (masked_td_error**2)).sum() / (mask[:, : -self.c_step].sum() + 1e-8)
        value_loss = (masked_td_error**2).sum() / (mask[:, : -self.c_step].sum() + 1e-8) # Original IQL loss
        loss = value_loss
        if (~th.isnan(loss)).all() and (~th.isinf(loss)).all():
            loss.backward()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("train_value/value_loss", value_loss.item(), t_env)
            # self.logger.log_stat("train_value/td_error_abs", masked_td_error.abs().mean().item(), t_env)
            # self.logger.log_stat("train_value/q_taken_mean", q_taken.mean().item(), t_env)
            # self.logger.log_stat("train_value/q_target_mean", q_target_next.mean().item(), t_env)
        if episode_num - self.last_target_update_episode >= self.args.target_update_interval:
            self._update_targets()
            self.last_target_update_episode = episode_num
            self.logger.console_logger.info(f"Updated target networks at episode {episode_num}")
        return value_loss

    def train_planner(self, batch: EpisodeBatch, t_env: int, episode_num: int, use_external_skill=False):
        # self.mac.agent.value.requires_grad_(False) # Handled by optimizer params

        rewards_batch = batch["reward"]
        actions_batch = batch["actions"]
        terminated_batch = batch["terminated"].float()
        mask_batch = batch["filled"].float()
        mask_batch[:, 1:] = mask_batch[:, 1:] * (1 - terminated_batch[:, :-1])
        
        b_size = batch.batch_size
        max_len = batch.max_seq_length

        self.mac.init_hidden(b_size)
        self.target_mac.init_hidden(b_size)

        accumulated_planner_internal_loss = th.tensor(0.0, device=self.device)
        accumulated_loss_dict = {}

        # Store skill-based predictions
        mac_skill_values = [] # Store skill-conditioned value predictions from planner's perspective
        mac_skill_rewards = [] # Store skill-conditioned reward predictions

        # --- Forward pass to get skills and planner's own losses ---
        for t in range(max_len - self.c_step):
            skill_input_actions = actions_batch[:, t] if use_external_skill else None
            
            # out_h_skill: (bs, entity_embed_dim) - the skill embedding from planner
            # planner_loss_step: sum of rec_loss, commit_loss, diver_loss from planner's VAE/VQ part
            # skill_idx: index from VQ if used
            # loss_dict_step: detailed dict of planner's internal losses
            out_h_skill, planner_loss_step, skill_idx, loss_dict_step = self.mac.forward_planner(
                batch, t, actions=actions_batch[:, t], training=True, loss_out=True, 
                skill_index_out=True, external_skill_index=skill_input_actions
            )
            accumulated_planner_internal_loss += planner_loss_step
            for key, value in loss_dict_step.items():
                accumulated_loss_dict[key] = accumulated_loss_dict.get(key, 0.0) + value

            # Use this skill to predict value and reward based on current state
            current_state_input = batch["state"][:, t] # (bs, state_dim)
            
            # value_pred_from_skill: (bs, entity_embed_dim) - embedding for value head
            value_pred_from_skill = self.mac.forward_planner_feedforward(
                out_h_skill, additional_input=current_state_input, forward_type="value"
            )[0] # Returns a list
            mac_skill_values.append(value_pred_from_skill)

            # reward_pred_from_skill: (bs, entity_embed_dim) - embedding for reward head
            reward_pred_from_skill = self.mac.forward_planner_feedforward(
                out_h_skill, additional_input=current_state_input, forward_type="reward"
            )[0]
            mac_skill_rewards.append(reward_pred_from_skill)

        # Handle last c_step for consistent list lengths if needed by value/reward prediction part
        # For simplicity, ensure mac_skill_values/rewards cover up to max_len if used for direct value prediction
        # The original ma_gumbel_learner fills these up to max_len
        for t_fill in range(max_len - self.c_step, max_len):
            # Simplified: repeat last skill or use a zero skill for padding if necessary
            # Or, run planner for these steps too if the logic requires it
             skill_input_actions_fill = actions_batch[:, t_fill] if use_external_skill else None
             out_h_skill_fill, _, _, _ = self.mac.forward_planner(
                batch, t_fill, actions=actions_batch[:, t_fill], training=True, loss_out=False, # No loss needed here
                skill_index_out=False, external_skill_index=skill_input_actions_fill
            )
             current_state_input_fill = batch["state"][:, t_fill]
             value_pred_from_skill_fill = self.mac.forward_planner_feedforward(
                out_h_skill_fill, additional_input=current_state_input_fill, forward_type="value"
            )[0]
             mac_skill_values.append(value_pred_from_skill_fill)
             reward_pred_from_skill_fill = self.mac.forward_planner_feedforward(
                out_h_skill_fill, additional_input=current_state_input_fill, forward_type="reward"
            )[0]
             mac_skill_rewards.append(reward_pred_from_skill_fill)


        # --- Value and Reward Prediction using generated skills ---
        # These are V(s_t | z_t) and R(s_t | z_t) where z_t is from planner(o_t, h_{t-1})
        value_predictions_skill_based = []  # Q-values from skill
        target_value_predictions_skill_based = []
        reward_predictions_skill_based = [] # Rewards from skill

        for t_pred in range(max_len):
            # mac.forward_value_skill takes (bs, skill_embedding_dim) and hidden_state (agent's value net hidden)
            # mac_skill_values[t_pred] is (bs, entity_embed_dim)
            # HISSDController.forward_value_skill expects (bs, concatenated_embeddings_from_list)
            # Let's adjust: mac_skill_values[t_pred] is already the embedding for the value head.
            # The HISSDController.forward_value_skill might need adaptation or this call needs to be to agent directly.
            # HISSDAgent.forward_value_skill(inputs, hidden_state) where inputs is the skill based emb.
            # For now, assume mac.forward_value_skill can take the (bs, entity_embed_dim) from mac_skill_values[t_pred]
            # This part needs careful alignment with HISSDController's methods.
            # HISSDController.forward_value_skill(bs, batch_emb_list) -> batch_emb = th.cat(batch_emb_list, dim=1)
            # So, we should pass a list: [mac_skill_values[t_pred]]
            
            # This part is tricky: ma_gumbel uses mac.forward_value(batch, t) for value_pre,
            # but it should be conditioned on the skill for planner training.
            # Let's use the skill-based value predictions.
            # The `forward_value_skill` in `HISSDAgent` is `self.value.forward_skill(inputs, hidden_state_value)`
            # `inputs` should be the skill-derived embedding. `mac_skill_values[t_pred]` is this.
            
            # We need agent-wise values here, then mix them.
            # The `mac_skill_values[t_pred]` is (bs, entity_embed_dim) - a global skill representation for value.
            # If ValueNet in HISSDAgent expects per-agent obs, this is not right.
            # ValueNet in HISSDAgent takes `input_shape` (obs).
            # `forward_value_skill` in HISSDAgent takes `inputs` (presumably skill-related) and `hidden_state`.
            # This suggests `inputs` to `forward_value_skill` should be shaped like what its internal transformer expects.
            
            # Re-think: The planner loss in ma_gumbel_learner uses `mixed_values` derived from `value_pre`.
            # `value_pre` comes from `self.mac.forward_value(batch, t=t, task=task)`. This is standard Q-learning value.
            # The skill `out_h` is used to predict `value_out_h` (an embedding), which is then *not* directly used to get Q-values for the TD error in ma_gumbel_learner's planner.
            # Instead, `mixed_values` (from standard Q) is used in TD error, and `value_out_h` (skill-based value embedding) is used in `mac_value.append(value_out_h)`.
            # This `mac_value` is then used in `mixed_values = self.mixer(value_pre, ...)` where `value_pre` is `mac_value`. This is confusing.

            # Let's follow ma_gumbel_learner's structure for `train_planner` more closely for TD error:
            # It uses standard Q-values for the TD error that drives the planner.
            current_q_values_std, _ = self.mac.agent.forward_value(self.mac._build_inputs(batch, t_pred), self.mac.hidden_states_value) # (bs*n_agents, 1)
            value_predictions_skill_based.append(current_q_values_std.reshape(b_size, self.n_agents, 1))
            with th.no_grad():
                target_q_values_std, _ = self.target_mac.agent.forward_value(self.target_mac._build_inputs(batch, t_pred), self.target_mac.hidden_states_value)
                target_value_predictions_skill_based.append(target_q_values_std.reshape(b_size, self.n_agents, 1))

            # Reward prediction
            # mac_skill_rewards[t_pred] is (bs, entity_embed_dim)
            # HISSDController.forward_reward_skill(bs, batch_emb_list)
            # HISSDAgent.forward_reward_skill(inputs, hidden_state_reward)
            # `inputs` here is the skill-derived embedding for reward.
            current_reward_pred, _ = self.mac.agent.forward_reward_skill(mac_skill_rewards[t_pred].unsqueeze(1), self.mac.hidden_states_reward) # Add sequence dim
            reward_predictions_skill_based.append(current_reward_pred) # (bs, 1)

        value_pre_stacked = th.stack(value_predictions_skill_based, dim=1) # (bs, T, n_agents, 1)
        target_value_pre_stacked = th.stack(target_value_predictions_skill_based, dim=1) # (bs, T, n_agents, 1)
        reward_pre_stacked = th.stack(reward_predictions_skill_based, dim=1) # (bs, T, 1)

        if self.mixer is not None:
            mixed_values_for_planner = self.mixer(value_pre_stacked, batch["state"])
            target_mixed_values_for_planner = self.target_mixer(target_value_pre_stacked, batch["state"]).detach()
        else:
            mixed_values_for_planner = value_pre_stacked.sum(dim=2)
            target_mixed_values_for_planner = target_value_pre_stacked.sum(dim=2).detach()
        
        # N-step rewards sum for TD target
        cs_rewards_sum = rewards_batch.clone() # (bs, T, 1)
        current_discount = self.args.gamma
        for i in range(1, self.c_step):
            cs_rewards_sum[:, :-(self.c_step)] += (current_discount**i) * rewards_batch[:, i : max_len - (self.c_step - i)]
        
        if self.adaptation:
            mask_sum_planner = mask_batch.sum()
            policy_loss = -mixed_values_for_planner.sum() / (mask_sum_planner + 1e-8)
        else:
            td_targets_planner = cs_rewards_sum[:, :-(self.c_step)] + \
                                (self.args.gamma**self.c_step) * \
                                (1 - terminated_batch[:, self.c_step:]) * \
                                target_mixed_values_for_planner[:, self.c_step:]
            
            td_error_planner = mixed_values_for_planner[:, :-(self.c_step)] - td_targets_planner.detach()
            masked_td_error_planner = td_error_planner * mask_batch[:, :-(self.c_step)]
            # policy_loss = -(masked_td_error_planner.mean()) # Maximize (Q - b), so minimize -(Q-b)
            # Changed to squared error to align with critic-style updates for planner's skill-values
            policy_loss = (masked_td_error_planner**2).sum() / (mask_batch[:, :-(self.c_step)].sum() + 1e-8)


        # Planner\'s own model loss (from VAE/VQ parts)
        num_planner_steps = max_len - self.c_step
        avg_planner_internal_loss = accumulated_planner_internal_loss / (num_planner_steps if num_planner_steps > 0 else 1e-8)

        # Reward prediction loss
        actual_rewards_for_pred = rewards_batch[:, :-(self.c_step)] # Align with predictions
        predicted_rewards = reward_pre_stacked[:, :-(self.c_step)]
        reward_pred_loss = F.mse_loss(predicted_rewards, actual_rewards_for_pred, reduction="none")
        reward_pred_loss = (reward_pred_loss * mask_batch[:, :-(self.c_step)]).sum() / (mask_batch[:, :-(self.c_step)].sum() + 1e-8)

        total_planner_loss = policy_loss + avg_planner_internal_loss + self.reward_loss_coef * reward_pred_loss
        
        if (~th.isnan(total_planner_loss)).all() and (~th.isinf(total_planner_loss)).all():
            total_planner_loss.backward()

        if episode_num - self.last_target_update_episode >= self.args.target_update_interval:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("train_planner/policy_loss", policy_loss.item(), t_env)
            self.logger.log_stat("train_planner/planner_internal_loss", avg_planner_internal_loss.item(), t_env)
            self.logger.log_stat("train_planner/reward_pred_loss", reward_pred_loss.item(), t_env)
            self.logger.log_stat("train_planner/total_loss", total_planner_loss.item(), t_env)
            if not self.adaptation:
                 self.logger.log_stat("train_planner/td_error_abs", masked_td_error_planner.abs().mean().item(), t_env)
            for key in accumulated_loss_dict:
                self.logger.log_stat(f"train_planner/internal_{key}", accumulated_loss_dict[key] / (max_len - self.c_step + 1e-8), t_env)
            self.log_stats_t = t_env # Update log time

        return total_planner_loss


    def pretrain(self, batch: EpisodeBatch, t_env: int, episode_num: int, use_external_skill=False):
        if self.pretrain_steps == 0:
            self.logger.console_logger.info("Starting Pretraining")
        
        self.zero_grad()
        vae_loss, act_rec_loss, inf_loss = self.train_vae(batch, t_env, episode_num, use_external_skill=use_external_skill)
        grad_norm = self.update(pretrain=True)
        self.pretrain_steps += 1
        
        if self.use_wandb and hasattr(self.args, 'wandb_detailed_logging') and self.args.wandb_detailed_logging:
            # wandb.log({
            #     "pretrain_step/vae_loss": vae_loss.item(), 
            #     "pretrain_step/act_rec_loss": act_rec_loss.item(),
            #     "pretrain_step/inf_loss": inf_loss.item(),
            #     "pretrain_step/grad_norm": grad_norm
            # }, step=t_env)
            pass


    def test_pretrain(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        self.test_vae(batch, t_env, episode_num)


    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, use_external_skill=False):
        if self.training_steps == 0:
            self.logger.console_logger.info("Starting Main Training")

        self.zero_grad() # Zero gradients for both optimizers if they share params, or manage separately
        
        value_loss = self.train_value(batch, t_env, episode_num)
        # self.update(pretrain=False) # Update after value loss

        # self.zero_grad() # Zero again if optimizers are separate or for clarity
        planner_loss = self.train_planner(batch, t_env, episode_num, use_external_skill=use_external_skill)
        grad_norm = self.update(pretrain=False) # Update after planner loss (and implicitly value if params shared)
        
        self.training_steps += 1

        if self.use_wandb and hasattr(self.args, 'wandb_detailed_logging') and self.args.wandb_detailed_logging:
            # wandb.log({
            #     "train_step/value_loss": value_loss.item(), 
            #     "train_step/planner_loss": planner_loss.item(),
            #     "train_step/grad_norm": grad_norm
            # }, step=t_env)
            pass


    def _update_targets(self):
        self.target_mac.load_state(self.mac) # HISSDController should implement load_state
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target networks.")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
        self.infonce_proj_skill.cuda()
        self.infonce_proj_act.cuda()
        self.device = 'cuda' # Ensure device is updated

    def save_models(self, path):
        self.mac.save_models(path) # HISSDController.save_models saves agent and optionally mixer
        # Optimiser state is for the combined parameters
        th.save(self.optimiser.state_dict(), f"{path}/opt.th")
        th.save(self.pre_optimiser.state_dict(), f"{path}/pre_opt.th")


    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path) # Also load target mac models
        
        opt_path = f"{path}/opt.th"
        pre_opt_path = f"{path}/pre_opt.th"
        
        try:
            self.optimiser.load_state_dict(
                th.load(opt_path, map_location=lambda storage, loc: storage)
            )
        except FileNotFoundError:
            self.logger.console_logger.warning(f"Optimizer state not found at {opt_path}")

        try:
            self.pre_optimiser.load_state_dict(
                th.load(pre_opt_path, map_location=lambda storage, loc: storage)
            )
        except FileNotFoundError:
             self.logger.console_logger.warning(f"Pre-Optimizer state not found at {pre_opt_path}")

