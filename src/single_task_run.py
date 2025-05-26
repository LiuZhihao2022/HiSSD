import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath
import copy
import json
import shutil
import wandb

# Assuming single-task components are compatible with these registry paths
# or are registered appropriately.
from learners import REGISTRY as le_REGISTRY
from runners.multi_task import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY

from components.episode_buffer import ReplayBuffer
from components.offline_buffer import OfflineBuffer
from components.transforms import OneHot
from mctx._src.network import ReplayBufferList
import numpy as np

from mctx._src.network import PolicyRNN
import logging
# 设置JAX相关的日志级别为INFO或更高(ERROR, CRITICAL)，这样DEBUG日志就不会显示
logging.getLogger('jax').setLevel(logging.INFO)

from collections import defaultdict
# Ensure MPEEnvWrapper and MPE runner are importable if MPE is used
from envs.mpe_env_wrapper import MPEEnvWrapper
# from runners.multi_task.mpe_hier_mcts_parallel_runner import MPEHierMCTSParallelRunner


class PrioritizedMultiTaskReplayBuffer:
    """为每个任务维护单独的ReplayBuffer，并支持优先级采样"""
    
    def __init__(self, task_list, task2buffer, batch_size, alpha=0.6):
        self.task_buffers = {}
        self.task_buffer_sizes = {} # Current number of episodes in each buffer
        self.task_max_buffer_sizes = {} # Max capacity of each buffer
        self.batch_size = batch_size    # 采样批次大小
        
        self.alpha = alpha              # 优先级指数
        self.priorities = {}            # 存储每个任务的所有样本优先级 (np.array per task)
        self.max_priorities = {}        # 每个任务的最大优先级
        
        for task in task_list:
            self.task_buffers[task] = task2buffer[task]
            self.task_max_buffer_sizes[task] = task2buffer[task].buffer_size
            self.task_buffer_sizes[task] = task2buffer[task].episodes_in_buffer
            self.priorities[task] = np.ones(self.task_max_buffer_sizes[task]) 
            self.max_priorities[task] = 1.0
    
    def insert_episode_batch(self, task, episode_batch):
        if task not in self.task_buffers:
            raise KeyError(f"Task {task} not found in PrioritizedMultiTaskReplayBuffer")
        
        buffer = self.task_buffers[task]
        max_buf_size = self.task_max_buffer_sizes[task]
        
        # buffer.buffer_index is the next slot to write to *before* insertion
        start_idx_in_buffer = buffer.buffer_index 
        
        buffer.insert_episode_batch(episode_batch) # This updates buffer.buffer_index and buffer.episodes_in_buffer
        
        num_inserted = episode_batch.batch_size
        
        current_idx = start_idx_in_buffer
        for _ in range(num_inserted):
            self.priorities[task][current_idx] = self.max_priorities[task]
            current_idx = (current_idx + 1) % max_buf_size
        
        self.task_buffer_sizes[task] = buffer.episodes_in_buffer

    def update_priorities(self, task, indices, new_priorities_values):
        if task not in self.priorities:
            # Log warning or raise error
            return
            
        for buf_idx, priority_val in zip(indices, new_priorities_values):
            # buf_idx is an actual index in the ReplayBuffer's internal flat array
            self.priorities[task][buf_idx] = priority_val
            
        self.max_priorities[task] = max(self.max_priorities[task], np.max(new_priorities_values))
    
    def can_sample(self, task, batch_size=None):
        if task not in self.task_buffers:
            return False
        
        if batch_size is None:
            batch_size = self.batch_size
            
        return self.task_buffer_sizes[task] >= batch_size
    
    def sample(self, task, batch_size=None):
        if task not in self.task_buffers:
            raise KeyError(f"Task {task} not found in PrioritizedMultiTaskReplayBuffer")
        
        if batch_size is None:
            batch_size = self.batch_size
            
        if not self.can_sample(task, batch_size):
            raise ValueError(f"Not enough episodes in buffer for task {task} to sample batch of size {batch_size}")
        
        buffer = self.task_buffers[task]
        num_valid_episodes = buffer.episodes_in_buffer
        max_buf_size = self.task_max_buffer_sizes[task]

        valid_indices_in_buffer_array = []
        if num_valid_episodes == max_buf_size: # Buffer is full, all indices are valid
            valid_indices_in_buffer_array = np.arange(max_buf_size)
        else: # Buffer not full, data is from 0 to num_valid_episodes-1 (assuming ReplayBuffer fills this way initially)
              # This part relies on ReplayBuffer.sample(bs, None) correctly handling its own circularity
              # to identify valid indices if we were to get them.
              # For PER, we need priorities of *specific* valid items.
              # A robust way: get all items from ReplayBuffer with their original indices, or ReplayBuffer exposes valid indices.
              # Assuming ReplayBuffer stores items densely from index 0 up to episodes_in_buffer if not full,
              # or if full, all indices 0 to max_buffer_size-1 are valid.
            if buffer.buffer_index <= num_valid_episodes and num_valid_episodes < max_buf_size : # Data is [0, num_valid_episodes-1]
                 valid_indices_in_buffer_array = np.arange(num_valid_episodes)
            else: # Buffer has wrapped or is full
                 valid_indices_in_buffer_array = np.arange(max_buf_size)


        priorities_for_sampling = self.priorities[task][valid_indices_in_buffer_array]
        
        probs = priorities_for_sampling ** self.alpha
        probs_sum = probs.sum()

        if probs_sum < 1e-6: # If all priorities are zero (or near zero)
            # Sample uniformly from the valid indices
            chosen_indices_relative_to_valid_array = np.random.choice(len(valid_indices_in_buffer_array), batch_size, replace=False)
        else:
            probs /= probs_sum
            chosen_indices_relative_to_valid_array = np.random.choice(len(valid_indices_in_buffer_array), batch_size, replace=False, p=probs)
        
        # Map these relative indices back to actual buffer indices
        actual_indices_to_sample = valid_indices_in_buffer_array[chosen_indices_relative_to_valid_array]
        
        episode_sample = buffer.sample(batch_size, actual_indices_to_sample)
        
        return episode_sample, actual_indices_to_sample # Return actual buffer indices

    def get_buffer_size(self, task): # Returns current number of episodes
        if task not in self.task_buffers:
            return 0
        return self.task_buffer_sizes[task]
    
    def get_total_size(self): # Returns total current episodes over all tasks
        return sum(self.task_buffer_sizes.values())


def get_single_task_name(args):
    if hasattr(args, "env_args") and "map_name" in args.env_args and args.env_args["map_name"] is not None:
        return args.env_args["map_name"]
    if hasattr(args, "env_args") and "scenario_name" in args.env_args and args.env_args["scenario_name"] is not None: # For MPE
        return args.env_args["scenario_name"]
    if hasattr(args, "scenario") and args.scenario is not None: # For SC2v2
        return args.scenario
    return "single_task"


def run(_run, _config, _log):
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config, indent=4, width=1)
    _log.info("\n\n" + experiment_params + "\n")

    results_save_dir = args.results_save_dir

    if args.use_tensorboard and not args.evaluate:
        tb_exp_direc = os.path.join(results_save_dir, "tb_logs")
        logger.setup_tb(tb_exp_direc)

    args.save_dir = os.path.join(results_save_dir, "models") # For offline models

    config_str = json.dumps(vars(args), indent=4)
    with open(os.path.join(results_save_dir, "config.json"), "w") as f:
        f.write(config_str)

    logger.setup_sacred(_run)

    run_single_task_pipeline(args=args, logger=logger)

    print("Exiting Main")
    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")
    print("Exiting script")
    os._exit(os.EX_OK)


def evaluate_single_task(main_args, logger, runner):
    n_test_runs = max(1, main_args.test_nepisode // main_args.batch_size_run)
    test_returns = []
    test_stats = defaultdict(list)

    with th.no_grad():
        for i in range(n_test_runs):
            stats = runner.run(test_mode=True, run_idx=i) # runner.run might return stats
            if stats is not None:
                if "episode_return" in stats: # Assuming runner.run can return this
                    test_returns.append(stats["episode_return"])
                for k,v in stats.items():
                    if isinstance(v, (int, float)):
                         test_stats[k].append(v)


        if main_args.save_replay:
            runner.save_replay()
        runner.close_env()

    avg_return = np.mean(test_returns) if test_returns else 0
    logger.log_stat("test_return_mean", avg_return, 0) # 0 for t_env placeholder
    for k,v_list in test_stats.items():
        logger.log_stat(f"test_{k}_mean", np.mean(v_list), 0)

    logger.print_recent_stats()


def init_single_task_components(main_args, logger):
    MAP_NAMES = {
        "terran_5_vs_5": "10gen_terran", "zerg_5_vs_5": "10gen_zerg", "terran_10_vs_10": "10gen_terran",
    }
    DISTRIBUTION_CONFIGS = {
        "terran_5_vs_5": {"n_units": 5, "n_enemies": 5, "team_gen": {"dist_type": "weighted_teams", "unit_types": ["marine", "marauder", "medivac"], "weights": [0.45, 0.45, 0.1], "observe": True}, "start_positions": {"dist_type": "surrounded_and_reflect", "p": 0.5, "n_enemies": 5, "map_x": 32, "map_y": 32}},
        "zerg_5_vs_5": {"n_units": 5, "n_enemies": 5, "team_gen": {"dist_type": "weighted_teams", "unit_types": ["zergling", "baneling", "hydralisk"], "weights": [0.45, 0.1, 0.45], "observe": True}, "start_positions": {"dist_type": "surrounded_and_reflect", "p": 0.5, "n_enemies": 5, "map_x": 32, "map_y": 32}},
        "terran_10_vs_10": {"n_units": 10, "n_enemies": 10, "team_gen": {"dist_type": "weighted_teams", "unit_types": ["marine", "marauder", "medivac"], "weights": [0.45, 0.45, 0.1], "observe": True}, "start_positions": {"dist_type": "surrounded_and_reflect", "p": 0.5, "n_enemies": 5, "map_x": 32, "map_y": 32}},
    }

    task_args = copy.deepcopy(main_args)
    task_name = get_single_task_name(task_args)

    if task_args.env == "sc2":
        if "map_name" not in task_args.env_args or task_args.env_args["map_name"] is None:
            logger.error("map_name not specified in env_args for SC2 environment!")
            raise ValueError("map_name required for SC2.")
    elif task_args.env == "sc2v2":
        if not hasattr(task_args, "scenario") or task_args.scenario not in MAP_NAMES:
            logger.error(f"Scenario {getattr(task_args, 'scenario', 'None')} not configured for SC2v2.")
            raise ValueError("Invalid scenario for SC2v2")
        task_args.env_args["map_name"] = MAP_NAMES[task_args.scenario]
        task_args.env_args["capability_config"] = DISTRIBUTION_CONFIGS[task_args.scenario]
    elif task_args.env == "mpe":
        if "scenario_name" not in task_args.env_args or task_args.env_args["scenario_name"] is None:
            logger.warning("scenario_name not specified in env_args for MPE. Using default or task_name.")
            task_args.env_args["scenario_name"] = task_name # Fallback
    
    runner = r_REGISTRY[main_args.runner](args=task_args, logger=logger, task=task_name)
    env_info = runner.get_env_info()
    for k, v in env_info.items():
        setattr(task_args, k, v)

    if getattr(task_args, "basic_action_as_skill", False) or task_args.env == "mpe":
        task_args.skill_dim = env_info["n_actions"]
        if hasattr(runner, 'args'): runner.args.skill_dim = task_args.skill_dim

    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "skills": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "avail_skills": {"vshape": (task_args.skill_dim,), "group": "agents", "dtype": th.int},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
        "episode_limit": {"vshape": (1,), "dtype": th.int},
    }
    if main_args.env in ["sc2", "sc2v2"]: scheme["reward"] = {"vshape": (1,)}
    elif main_args.env == "mpe": scheme["reward"] = {"vshape": (1,)} if main_args.common_reward else {"vshape": (env_info["n_agents"],)}
    elif main_args.common_reward: scheme["reward"] = {"vshape": (1,)}
    else: scheme["reward"] = {"vshape": (main_args.n_agents,)}

    groups = {"agents": env_info["n_agents"]}
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=task_args.n_actions)]),
        "skills": ("skills_onehot", [OneHot(out_dim=task_args.skill_dim)])
    }

    replay_buffer = ReplayBuffer(
        scheme, groups, task_args.buffer_size, env_info["episode_limit"] + 1,
        preprocess=preprocess, device="cpu" if task_args.buffer_cpu_only else task_args.device,
    )
    return task_args, runner, replay_buffer, scheme, groups, preprocess, task_name


def train_offline_phase(
    main_args, logger, learner, task_args, runner, offline_buffer,
    t_start=0, pretrain=False
):
    t_env = t_start
    episode = 0 
    t_max = main_args.t_max if not pretrain else main_args.pretrain_steps
    model_save_time = 0
    last_test_T = t_start
    last_log_T = t_start
    start_time = time.time()
    last_time = start_time
    test_time_total = 0 # Not used actively in this simplified version's loop

    batch_size_train = main_args.batch_size
    batch_size_run = main_args.batch_size_run # For episode count increment

    while t_env < t_max:
        episode_sample = offline_buffer.sample(batch_size_train)
        if episode_sample.device != task_args.device:
            episode_sample.to(task_args.device)

        terminated_by_learner = False
        if pretrain:
            if hasattr(learner, "pretrain"):
                # Assuming learner.pretrain does not signal termination for fixed steps
                learner.pretrain(episode_sample, t_env, episode, use_external_skill=getattr(main_args, "pretrain_use_external_skill", False))
            else:
                raise ValueError("Learner does not have a `pretrain` method!")
        else:
            # Assuming learner.train does not signal termination
            learner.train(episode_sample, t_env, episode, use_external_skill=getattr(main_args, "train_use_external_skill", False))

        if terminated_by_learner: # If learner could terminate
            logger.console_logger.info(f"Training terminated by learner at t_env = {t_env}.")
            break
        
        t_env += 1
        episode += batch_size_run 

        if (t_env - last_test_T) / main_args.test_interval >= 1.0 or t_env >= t_max:
            # In-loop testing can be added here if needed, using `runner`
            # Ensure `runner.mac` is updated if `learner.mac` has changed.
            # For simplicity, this example relies on separate evaluation calls.
            current_stage_name = "Pretraining" if pretrain else "Offline Training"
            logger.console_logger.info(f"{current_stage_name} Step: {t_env} / {t_max}")
            logger.console_logger.info(
                "Estimated time left: {}. Time passed: {}".format(
                    time_left(last_time, last_test_T, t_env, t_max),
                    time_str(time.time() - start_time)
                )
            )
            last_time = time.time()
            last_test_T = t_env

        if main_args.save_model and (t_env - model_save_time >= main_args.save_model_interval or model_save_time == 0 or t_env >= t_max):
            save_path_prefix = getattr(main_args, "pretrain_save_dir", os.path.join(main_args.results_save_dir, "pretrain_models")) if pretrain else main_args.save_dir
            save_path = os.path.join(save_path_prefix, str(t_env))
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info(f"Saving models to {save_path}")
            learner.save_models(save_path)
            model_save_time = t_env
        
        if (t_env - last_log_T) >= main_args.log_interval:
            logger.log_stat("episode", episode, t_env)
            logger.print_recent_stats()
            last_log_T = t_env


def train_online_mcts_phase(
    main_args, logger, mac, learner, task_args,
    scheme, groups, preprocess, # For MCTS runner setup
    online_learner_buffer, # ReplayBuffer instance for learner
    task_name, t_start=0
):
    model_save_time = 0
    last_test_T = t_start 
    last_log_T = t_start
    last_mcts_target_update_episode = 0 # Based on learner's episode view
    start_time = time.time()
    last_time = start_time
    test_time_total = 0
    
    mcts_replay_buffer_list = ReplayBufferList(
        capacity=getattr(main_args, "replay_buffer_list_capacity", 100), use_real_data=True
    )

    use_priority_sampling = getattr(main_args, "use_priority_sampling", False)
    # `online_learner_buffer` is the ReplayBuffer. If PER, wrap it.
    if use_priority_sampling:
        # This re-assigns online_learner_buffer to the prioritized wrapper
        online_learner_buffer = PrioritizedMultiTaskReplayBuffer(
            task_list=[task_name], task2buffer={task_name: online_learner_buffer},
            batch_size=main_args.batch_size, alpha=getattr(main_args, "per_alpha", 0.6)
        )

    mcts_target_update_interval = getattr(main_args, "mcts_target_update_interval", 80)
    batch_size_run = main_args.batch_size_run
    batch_size_train_learner = getattr(main_args, "batch_size_train_learner", 32)
    batch_size_train_mcts = getattr(main_args, "mcts_batch_size", main_args.batch_size)

    test_nepisode_online = getattr(main_args, "test_nepisode_online", main_args.test_nepisode)
    test_interval_online = getattr(main_args, "test_interval_online", main_args.test_interval)
    n_test_runs = max(1, test_nepisode_online // batch_size_run)
    if main_args.evaluate or main_args.debug: n_test_runs = 0

    mcts_runner_key = main_args.mcts_runner
    if task_args.env == "mpe" and hasattr(main_args, "mpe_mcts_runner"):
        mcts_runner_key = main_args.mpe_mcts_runner
    
    mcts_runner = r_REGISTRY[mcts_runner_key](args=task_args, logger=logger, task=task_name)
    mcts_runner.t_env = t_start

    env_info_mcts = mcts_runner.get_env_info()
    state_shape = env_info_mcts["state_shape"]
    obs_shape = env_info_mcts["obs_shape"]
    n_actions_env = env_info_mcts["n_actions"]
    
    mcts_skill_dim = task_args.skill_dim
    if getattr(task_args, "basic_action_as_skill", False) or task_args.env == "mpe":
        mcts_skill_dim = n_actions_env
        if task_args.skill_dim != mcts_skill_dim: task_args.skill_dim = mcts_skill_dim

    n_agents = env_info_mcts["n_agents"]
    input_shape_mcts = obs_shape
    if task_args.obs_last_action: input_shape_mcts += mcts_skill_dim
    if task_args.obs_agent_id: input_shape_mcts += n_agents

    mcts_network = PolicyRNN(
        obs_input_shape=input_shape_mcts, emb_input_shape=state_shape, output_shape=mcts_skill_dim,
        num_agents=n_agents, c_step=mac.c_step, device=main_args.device,
        optimizer=getattr(main_args, "mcts_optimizer", 'adam'),
        hidden_dim=getattr(main_args, "mcts_hidden_dim", 128),
        seed=getattr(main_args, "seed", 42)
    )
    mcts_network.init_hidden(batch_size=batch_size_run)

    if getattr(main_args, "mcts_network_path", "") != "":
        mcts_load_path = os.path.join(main_args.mcts_network_path, "mcts_network.th")
        mcts_network.load_state_dict(th.load(mcts_load_path, map_location=main_args.device))
        mcts_network.update_target_network() # Ensure target is also updated
        logger.console_logger.info(f"Loaded MCTS network from {mcts_load_path}")

    mcts_runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac, mcts_network=mcts_network)

    logger.console_logger.info("Beginning online MCTS training stage...")
    current_t_env = t_start
    episode_count_learner = 0 
    use_wandb = getattr(main_args, "use_wandb", False) # From main_args

    train_stats_window = defaultdict(lambda: defaultdict(lambda: [])) # For windowed averages
    window_size = getattr(main_args, "train_stats_window_size", 20)


    while current_t_env < main_args.online_steps:
        episode_batch_collected, mcts_sample_for_replay, stats_info = mcts_runner.run(test_mode=False)

        if use_priority_sampling:
            online_learner_buffer.insert_episode_batch(task_name, episode_batch_collected)
        else:
            online_learner_buffer.insert_episode_batch(episode_batch_collected)
        if use_priority_sampling:
            can_sample_learner = online_learner_buffer.can_sample(task_name if use_priority_sampling else None, batch_size_train_learner)
        else:
            can_sample_learner = online_learner_buffer.can_sample(batch_size_train_learner)

        if can_sample_learner:
            training_batch_learner, indices_learner = (None, None)
            if use_priority_sampling:
                training_batch_learner, indices_learner = online_learner_buffer.sample(task_name, batch_size_train_learner)
            else:
                training_batch_learner = online_learner_buffer.sample(batch_size_train_learner)
            
            if training_batch_learner.device != main_args.device: training_batch_learner.to(main_args.device)
            
            # Assuming HISSDLearner.train can return losses for PER
            # learner_losses_dict = learner.train(training_batch_learner, current_t_env, episode_count_learner, use_external_skill=True, return_losses=True) 
            # TODO: train value now but not planner
            value_loss = learner.train_value(training_batch_learner, current_t_env, episode_count_learner)
            if use_wandb:
                wandb.log({
                    "learner/value_loss": value_loss.item(),
                }, step=current_t_env)
            # if use_priority_sampling and indices_learner is not None and learner_losses_dict is not None:
            #     # Example: use 'value_loss_per_item' if learner returns it. Placeholder for now.
            #     # td_errors_or_losses = learner_losses_dict.get('td_error', np.ones(len(indices_learner)))
            #     # priorities_new = (np.abs(td_errors_or_losses) + getattr(main_args, "per_epsilon", 1e-6))
            #     # Fallback: use a fixed high priority for new samples if detailed losses not available
            #     priorities_new = np.full(len(indices_learner), online_learner_buffer.max_priorities[task_name])
            #     value_loss = learner_losses_dict.get('value_loss', None) # This is likely a scalar
            #     if isinstance(value_loss, th.Tensor) and value_loss.numel() == 1: # if scalar loss
            #          priorities_new = np.full(len(indices_learner), value_loss.item() + 1e-6) # Example
            #     elif isinstance(value_loss, (list, np.ndarray)) and len(value_loss) == len(indices_learner): # if per-sample
            #          priorities_new = np.abs(value_loss) + 1e-6

            #     online_learner_buffer.update_priorities(task_name, indices_learner, priorities_new)

        if mcts_sample_for_replay is not None: mcts_replay_buffer_list.push(mcts_sample_for_replay)
        
        current_t_env = mcts_runner.t_env 
        episode_count_learner += batch_size_run

        # Log training run stats
        if stats_info and use_wandb:
            log_data_train = {}
            for key, value in stats_info.items():
                if isinstance(value, (int, float)):
                    log_data_train[f"train_online/{task_name}/{key}"] = value
                    # Update window
                    train_stats_window[task_name][key].append(value)
                    if len(train_stats_window[task_name][key]) > window_size:
                        train_stats_window[task_name][key].pop(0)
                    log_data_train[f"train_online/{task_name}/{key}_window_avg"] = np.mean(train_stats_window[task_name][key])
            if log_data_train: wandb.log(log_data_train, step=current_t_env)


        if len(mcts_replay_buffer_list) >= batch_size_train_mcts:
            mcts_batch_sample = mcts_replay_buffer_list.sample(batch_size_train_mcts)
            loss_dict_mcts = mcts_network.train_network(
                batch=mcts_batch_sample, gamma=getattr(main_args, "mcts_gamma", 0.99),
                value_loss_weight=getattr(main_args, "mcts_value_loss_weight", 0.5),
                max_grad_norm=getattr(main_args, "mcts_max_grad_norm", 10.0), use_real_data=False
            )
            if use_wandb:
                wandb.log({
                    "mcts/total_loss": loss_dict_mcts.get("total_loss",0),
                    "mcts/policy_loss": loss_dict_mcts.get("policy_loss",0),
                    "mcts/value_loss": loss_dict_mcts.get("value_loss",0),
                    "mcts/entropy": loss_dict_mcts.get("entropy",0),
                    "mcts/buffer_size": len(mcts_replay_buffer_list),
                }, step=current_t_env)
            for k,v in loss_dict_mcts.items(): logger.log_stat(f"mcts_{k}", v, current_t_env)


            if (episode_count_learner - last_mcts_target_update_episode) / mcts_target_update_interval >= 1.0:
                mcts_network.update_target_network()
                last_mcts_target_update_episode = episode_count_learner
                logger.console_logger.info(f"Updated MCTS target network at t_env: {current_t_env}")

        if (current_t_env - last_test_T) / test_interval_online >= 1.0 or current_t_env >= main_args.online_steps:
            test_start_time_iter = time.time()
            # Simplified test logging
            # Actual testing would run mcts_runner in test_mode for n_test_runs
            # And aggregate stats similar to evaluate_single_task
            logger.console_logger.info(f"Online Step: {current_t_env} / {main_args.online_steps}")
            logger.console_logger.info(
                "Est time left: {}. Passed: {}. Test time: {}".format(
                    time_left(last_time, last_test_T, current_t_env, main_args.online_steps),
                    time_str(time.time() - start_time), time_str(time.time() - test_start_time_iter)
                )
            )
            test_time_total += time.time() - test_start_time_iter
            last_time = time.time()
            last_test_T = current_t_env

        save_interval_online = getattr(main_args, "save_model_interval_online", main_args.save_model_interval)
        if main_args.save_model and (current_t_env - model_save_time >= save_interval_online or model_save_time == 0 or current_t_env >= main_args.online_steps):
            online_save_dir = getattr(main_args, "online_save_dir") # Should be set
            save_path_online = os.path.join(online_save_dir, str(current_t_env))
            os.makedirs(save_path_online, exist_ok=True)
            logger.console_logger.info(f"Saving online models to {save_path_online}")
            learner.save_models(save_path_online)
            th.save(mcts_network.state_dict(), os.path.join(save_path_online, "mcts_network.th"))
            model_save_time = current_t_env

        if (current_t_env - last_log_T) >= main_args.log_interval:
        # if True:
            # Ensure episode is logged before printing stats
            logger.log_stat("episode", episode_count_learner, current_t_env)
            logger.log_stat("episode_learner", episode_count_learner, current_t_env) # Keep this for additional tracking if needed
            logger.print_recent_stats()
            last_log_T = current_t_env
            
    # Final save
    if main_args.save_model:
        online_save_dir = getattr(main_args, "online_save_dir")
        save_path_final_online = os.path.join(online_save_dir, "final")
        os.makedirs(save_path_final_online, exist_ok=True)
        logger.console_logger.info(f"Saving final online models to {save_path_final_online}")
        learner.save_models(save_path_final_online)
        th.save(mcts_network.state_dict(), os.path.join(save_path_final_online, "mcts_network.th"))
    
    logger.console_logger.info("Finished online MCTS training")


def run_single_task_pipeline(args, logger):
    main_args = copy.deepcopy(args)

    task_args, runner, replay_buffer, scheme, groups, preprocess, task_name = \
        init_single_task_components(main_args, logger)

    if getattr(main_args, "basic_action_as_skill", False) and not hasattr(task_args, "skill_dim"):
         task_args.skill_dim = task_args.n_actions
    main_args = task_args
    mac = mac_REGISTRY[main_args.mac](scheme=replay_buffer.scheme, args=task_args)
    runner.setup(scheme=replay_buffer.scheme, groups=groups, preprocess=preprocess, mac=mac)
    learner = le_REGISTRY[main_args.learner](mac, logger, main_args)

    if main_args.use_cuda:
        learner.cuda()

    if main_args.checkpoint_path != "":
        timesteps = []
        timestep_to_load = 0
        if not os.path.isdir(main_args.checkpoint_path):
            logger.console_logger.info(f"Checkpoint directory {main_args.checkpoint_path} doesn't exist")
            return
        for name in os.listdir(main_args.checkpoint_path):
            full_name = os.path.join(main_args.checkpoint_path, name)
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))
        if not timesteps:
            logger.console_logger.info(f"No valid timesteps found in {main_args.checkpoint_path}")
            return
            
        timestep_to_load = max(timesteps) if main_args.load_step == 0 else min(timesteps, key=lambda x: abs(x - main_args.load_step))
        model_path = os.path.join(main_args.checkpoint_path, str(timestep_to_load))
        logger.console_logger.info(f"Loading model from {model_path}")
        learner.load_models(model_path) # Learner loads MAC and its own state

        if main_args.evaluate or main_args.save_replay:
            # Ensure runner's MAC is the loaded one
            runner.mac = learner.mac # Or runner.setup(mac=learner.mac, ...)
            evaluate_single_task(main_args, logger, runner)
            return

    if getattr(main_args, "pretrain", False):
        pretrain_data_quality_config = getattr(main_args, "pretrain_data_quality", "medium")
        pretrain_data_quality = pretrain_data_quality_config.get(task_name, pretrain_data_quality_config) if isinstance(pretrain_data_quality_config, dict) else pretrain_data_quality_config
        
        pretrain_offline_buffer = OfflineBuffer(
            task_name, pretrain_data_quality,
            data_folder=main_args.offline_data_name, 
            dataset_folder=getattr(main_args, "offline_data_folder", "datasets"),
            offline_data_size=args.offline_data_size, random_sample=args.offline_data_shuffle,
        )
        logger.console_logger.info(f"Beginning pre-training with {main_args.pretrain_steps} timesteps.")
        train_offline_phase(main_args, logger, learner, task_args, runner, pretrain_offline_buffer, pretrain=True)
        
        logger.console_logger.info("Finished pretraining.")
        pretrain_save_dir = getattr(main_args, "pretrain_save_dir", os.path.join(main_args.results_save_dir, "pretrain_models"))
        os.makedirs(pretrain_save_dir, exist_ok=True)
        save_path_pretrain = os.path.join(pretrain_save_dir, str(main_args.pretrain_steps))
        logger.console_logger.info(f"Saving pretrained models to {save_path_pretrain}")
        learner.save_models(save_path_pretrain)

    if getattr(main_args, "load_wm_path", "") != "":
        logger.console_logger.info(f"Loading world model from {main_args.load_wm_path}")
        learner.load_models(main_args.load_wm_path)
        if getattr(main_args, "test_offline_model", False):
            runner.mac = learner.mac # Update runner's mac
            evaluate_single_task(main_args, logger, runner)
            return
        
    if getattr(main_args, "use_offline_training", False):
        train_data_quality_config = getattr(main_args, "train_data_quality", "medium")
        train_data_quality = train_data_quality_config.get(task_name, train_data_quality_config) if isinstance(train_data_quality_config, dict) else train_data_quality_config

        main_offline_buffer = OfflineBuffer(
            task_name, train_data_quality,
            data_folder=main_args.offline_data_name,
            dataset_folder=getattr(main_args, "offline_data_folder", "datasets"),
            offline_data_size=args.offline_data_size, random_sample=args.offline_data_shuffle,
        )
        logger.console_logger.info(f"Beginning offline training with {main_args.t_max} timesteps.")

        train_offline_phase(main_args, logger, learner, task_args, runner, main_offline_buffer, pretrain=False)
        
        if main_args.save_model:
            save_path_offline = os.path.join(main_args.save_dir, str(main_args.t_max)) # main_args.save_dir is for offline models
            os.makedirs(save_path_offline, exist_ok=True)
            logger.console_logger.info(f"Saving final offline models to {save_path_offline}")
            learner.save_models(save_path_offline)

    if getattr(main_args, "use_online_mcts", False):
        online_save_dir = os.path.join(main_args.results_save_dir, "online_models")
        os.makedirs(online_save_dir, exist_ok=True)
        main_args.online_save_dir = online_save_dir 

        main_args.online_steps = getattr(main_args, "online_steps", main_args.t_max)
        
        # replay_buffer from init_single_task_components is used as the base for learner's online data
        train_online_mcts_phase(
            main_args, logger, mac, learner, task_args,
            scheme, groups, preprocess, replay_buffer, task_name, t_start=0
        )
    else:
        runner.close_env()

    logger.console_logger.info("Finished Training Pipeline.")


def args_sanity_check(config, _log):
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF as no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"] // config["batch_size_run"]) * config["batch_size_run"]
    return config

# Note: If this script is to be run directly (e.g. with sacred),
# the typical `if __name__ == '__main__':` block with sacred setup would be needed.
# For now, assuming it's called via a main experiment script like the original.
