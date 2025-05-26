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
import wandb  # 添加wandb导入

from learners.multi_task import REGISTRY as le_REGISTRY
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

# 添加所需的导入
from collections import defaultdict
from envs.mpe_env_wrapper import MPEEnvWrapper # Ensure MPEEnvWrapper is importable
from runners.multi_task.mpe_hier_mcts_parallel_runner import MPEHierMCTSParallelRunner # Ensure MPE runner is importable

# 将MultiTaskReplayBuffer类升级为支持优先级的版本
class PrioritizedMultiTaskReplayBuffer:
    """为每个任务维护单独的ReplayBuffer，并支持优先级采样"""
    
    def __init__(self, task_list, task2buffer, batch_size, alpha=0.6):
        self.task_buffers = {}
        self.task_buffer_sizes = {}
        self.batch_size = batch_size    # 采样批次大小
        
        # 优先级采样参数
        self.alpha = alpha              # 优先级指数
        self.priorities = {}            # 存储每个任务的所有样本优先级
        self.max_priorities = {}        # 每个任务的最大优先级
        
        # 使用已有的buffer而不是创建新的
        for task in task_list:
            # 直接引用task2buffer中的buffer
            self.task_buffers[task] = task2buffer[task]
            self.buffer_size = task2buffer[task].buffer_size
            self.task_buffer_sizes[task] = task2buffer[task].episodes_in_buffer
            self.priorities[task] = np.ones(self.buffer_size)  # 初始优先级全为1
            self.max_priorities[task] = 1.0
    
    def insert_episode_batch(self, task, episode_batch):
        """将episode_batch插入到对应任务的buffer中，并设置优先级"""
        if task not in self.task_buffers:
            raise KeyError(f"Task {task} not found in PrioritizedMultiTaskReplayBuffer")
        
        # 获取当前任务的buffer和状态
        buffer = self.task_buffers[task]
        buffer_index = buffer.buffer_index
        buffer_size = buffer.buffer_size
        
        # 插入前记录当前buffer_index以便更新优先级
        start_idx = buffer_index
        
        # 插入数据到ReplayBuffer
        buffer.insert_episode_batch(episode_batch)
        
        # 更新优先级 - 需要考虑循环缓冲区的情况
        batch_size = episode_batch.batch_size
        
        # 计算插入后的buffer_index（考虑循环）
        new_buffer_index = (start_idx + batch_size) % buffer_size
        
        # 根据插入情况更新优先级
        if start_idx + batch_size <= buffer_size:
            # 简单情况 - 不需要循环回到缓冲区开始
            for i in range(batch_size):
                idx = start_idx + i
                self.priorities[task][idx] = self.max_priorities[task]
        else:
            # 复杂情况 - 需要在到达缓冲区末尾回到开始
            # 第一部分：从start_idx到buffer_size-1
            for i in range(buffer_size - start_idx):
                idx = start_idx + i
                self.priorities[task][idx] = self.max_priorities[task]
                
            # 第二部分：从0到new_buffer_index-1
            for i in range(new_buffer_index):
                self.priorities[task][i] = self.max_priorities[task]
        
        # 更新任务缓冲区的大小
        self.task_buffer_sizes[task] = buffer.episodes_in_buffer
    
    def update_priorities(self, task, indices, priorities):
        """更新指定任务中指定样本的优先级"""
        if task not in self.priorities:
            return
            
        for idx, priority in zip(indices, priorities):
            # 直接使用模运算转换索引
            buffer_idx = idx % self.buffer_size
            self.priorities[task][buffer_idx] = priority
            
        self.max_priorities[task] = max(self.max_priorities[task], np.max(priorities))
    
    def can_sample(self, task, batch_size=None):
        """检查指定任务的buffer是否可以采样"""
        if task not in self.task_buffers:
            return False
        
        if batch_size is None:
            batch_size = self.batch_size
            
        return self.task_buffer_sizes[task] >= batch_size
    
    def sample(self, task, batch_size=None):
        """从指定任务的buffer中基于优先级采样数据"""
        if task not in self.task_buffers:
            raise KeyError(f"Task {task} not found in PrioritizedMultiTaskReplayBuffer")
        
        if batch_size is None:
            batch_size = self.batch_size
            
        if not self.can_sample(task, batch_size):
            raise ValueError(f"Not enough episodes in buffer for task {task} to sample batch of size {batch_size}")
        
        # 获取当前有效的样本数量
        actual_size = min(self.task_buffer_sizes[task], self.buffer_size)
        
        # 计算采样概率
        priorities = self.priorities[task][:actual_size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # 采样索引
        indices = np.random.choice(actual_size, batch_size, replace=False, p=probs)
        
        # 从原始ReplayBuffer采样
        episode_sample = self.task_buffers[task].sample(batch_size, indices)
        
        # 返回样本和相关信息
        return episode_sample, indices
    
    def get_buffer_size(self, task):
        """获取指定任务buffer中的episode数量"""
        if task not in self.task_buffers:
            return 0
        return self.task_buffer_sizes[task]
    
    def get_total_size(self):
        """获取所有任务buffer中的episode总数"""
        return sum(self.task_buffer_sizes.values())

def run(_run, _config, _log):
    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config, indent=4, width=1)
    _log.info("\n\n" + experiment_params + "\n")

    results_save_dir = args.results_save_dir

    if args.use_tensorboard and not args.evaluate:
        # only log tensorboard when in training mode
        # though we are always in training mode when we reach here
        tb_exp_direc = os.path.join(results_save_dir, "tb_logs")
        logger.setup_tb(tb_exp_direc)

    # set model save dir
    args.save_dir = os.path.join(results_save_dir, "models")

    # write config file
    config_str = json.dumps(vars(args), indent=4)
    with open(os.path.join(results_save_dir, "config.json"), "w") as f:
        f.write(config_str)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def evaluate_sequential(main_args, logger, task2runner):
    n_test_runs = max(1, main_args.test_nepisode // main_args.batch_size_run)
    with th.no_grad():
        for task in main_args.test_tasks:
            for _ in range(n_test_runs):
                task2runner[task].run(test_mode=True)

            if main_args.save_replay:
                task2runner[task].save_replay()

            task2runner[task].close_env()

    logger.log_stat("episode", 0, 0)
    logger.print_recent_stats()


def init_tasks(task_list, main_args, logger):
    # 只在此处定义
    MAP_NAMES = {
        "terran_5_vs_5": "10gen_terran",
        "zerg_5_vs_5": "10gen_zerg",
        "terran_10_vs_10": "10gen_terran",
    }
    DISTRIBUTION_CONFIGS = {
        "terran_5_vs_5": {
            "n_units": 5,
            "n_enemies": 5,
            "team_gen": {
                "dist_type": "weighted_teams",
                "unit_types": ["marine", "marauder", "medivac"],
                "exception_unit_types": ["baneling"],
                "weights": [0.45, 0.45, 0.1],
                "observe": True,
            },
            "start_positions": {
                "dist_type": "surrounded_and_reflect",
                "p": 0.5,
                "n_enemies": 5,
                "map_x": 32,
                "map_y": 32,
            },
        },
        "zerg_5_vs_5": {
            "n_units": 5,
            "n_enemies": 5,
            "team_gen": {
                "dist_type": "weighted_teams",
                "unit_types": ["zergling", "baneling", "hydralisk"],
                "exception_unit_types": ["baneling"],
                "weights": [0.45, 0.1, 0.45],
                "observe": True,
            },
            "start_positions": {
                "dist_type": "surrounded_and_reflect",
                "p": 0.5,
                "n_enemies": 5,
                "map_x": 32,
                "map_y": 32,
            },
        },
        "terran_10_vs_10": {
            "n_units": 10,
            "n_enemies": 10,
            "team_gen": {
                "dist_type": "weighted_teams",
                "unit_types": ["marine", "marauder", "medivac"],
                "exception_unit_types": ["baneling"],
                "weights": [0.45, 0.45, 0.1],
                "observe": True,
            },
            "start_positions": {
                "dist_type": "surrounded_and_reflect",
                "p": 0.5,
                "n_enemies": 5,
                "map_x": 32,
                "map_y": 32,
            },
        },
    }

    task2args, task2runner, task2buffer = {}, {}, {}
    task2scheme, task2groups, task2preprocess = {}, {}, {}

    for task in task_list:
        task_args = copy.deepcopy(main_args)
        # 统一设置map_name和capability_config
        if task_args.env == "sc2":
            task_args.env_args["map_name"] = task
        elif task_args.env == "sc2v2":
            task_args.env_args["map_name"] = MAP_NAMES[task_args.scenario]
            task_args.env_args["capability_config"] = DISTRIBUTION_CONFIGS[task_args.scenario]
        elif task_args.env == "gymma":
            task_args.env_args["N"] = task
        elif task_args.env == "mpe":  # 为MPE环境添加处理
            task_args.env_args["scenario_name"] = task  # 假设 'task' 是MPE场景的名称
            # MPE specific args can be added here if needed, e.g.
            # task_args.env_args["continuous_actions"] = False # Or True, depending on MPE wrapper
        task2args[task] = task_args
        # Runner initialization will use the runner type specified in main_args (e.g., main_args.runner or main_args.mcts_runner)
        # For MPE, ensure the correct runner (like MPEHierMCTSParallelRunner) is specified in the config.
        task_runner = r_REGISTRY[main_args.runner](
            args=task_args, logger=logger, task=task
        )
        task2runner[task] = task_runner

        env_info = task_runner.get_env_info()
        for k, v in env_info.items():
            setattr(task_args, k, v)
        
        # skill_dim might be set based on n_actions for MPE if basic actions are skills
        if getattr(task_args, "basic_action_as_skill", False) or task_args.env == "mpe": # Default for MPE
            task_args.skill_dim = env_info["n_actions"] # Assuming MPE wrapper provides discrete n_actions
            if hasattr(task2runner[task], 'args'): # Ensure runner has args attribute
                 task2runner[task].args.skill_dim = task_args.skill_dim


        # scheme适配SMACv2
        scheme = {
            "state": {"vshape": env_info["state_shape"]},
            "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
            "skills": {"vshape": (1,), "group": "agents", "dtype": th.long}, # If using skills
            "avail_actions": {
                "vshape": (env_info["n_actions"],),
                "group": "agents",
                "dtype": th.int,
            },
            "avail_skills": { # If using skills
                "vshape": (task_args.skill_dim,),
                "group": "agents",
                "dtype": th.int,
            },
            "terminated": {"vshape": (1,), "dtype": th.uint8},
            "episode_limit": {"vshape": (1,), "dtype": th.int}, # MPE might not have this, ensure wrapper provides
        }
        # SMACv2奖励结构适配
        if main_args.env in ["sc2", "sc2v2"]:
            scheme["reward"] = {"vshape": (1,)}
        elif main_args.env == "mpe": # MPE reward structure
            if main_args.common_reward: # Or a specific arg like main_args.mpe_common_reward
                scheme["reward"] = {"vshape": (1,)}
            else: # Per-agent rewards
                scheme["reward"] = {"vshape": (env_info["n_agents"],)}
        elif main_args.common_reward:
            scheme["reward"] = {"vshape": (1,)}
        else:
            scheme["reward"] = {"vshape": (main_args.n_agents,)} # Fallback, ensure n_agents is correct
        
        groups = {"agents": env_info["n_agents"]} # Use n_agents from env_info for MPE
        preprocess = {
            "actions": ("actions_onehot", [OneHot(out_dim=task_args.n_actions)]),
            "skills": ("skills_onehot", [OneHot(out_dim=task_args.skill_dim)])
        }
        # preprocess = dict(preprocess_list)


        task2buffer[task] = ReplayBuffer(
            scheme,
            groups,
            task_args.buffer_size,
            env_info["episode_limit"] + 1,
            preprocess=preprocess,
            device="cpu" if task_args.buffer_cpu_only else task_args.device,
        )

        task2scheme[task], task2groups[task], task2preprocess[task] = (
            scheme,
            groups,
            preprocess,
        )

    # # Initialize MPE tasks if specified
    # if "mpe" in main_args.task_type:
    #     logger.info(f"Initializing MPE tasks...")
    #     mpe_scenario_name = getattr(main_args, "mpe_scenario_name", "simple_spread_v3") # Default MPE scenario
        
    #     # Env args for MPE
    #     mpe_env_args = {
    #         "scenario_name": mpe_scenario_name,
    #         "episode_limit": getattr(main_args, "mpe_episode_limit", 25), # Default episode limit for MPE
    #         "seed": main_args.seed,
    #         # Add any other MPE specific env_args from args
    #         "continuous_actions": getattr(main_args, "mpe_continuous_actions", False), # Example
    #         "max_cycles": getattr(main_args, "mpe_episode_limit", 25), # PettingZoo uses max_cycles
    #     }
    #     if hasattr(main_args, "common_reward"): # If common_reward is a global arg
    #          mpe_env_args["common_reward"] = main_args.common_reward

    #     # Create a temporary MPE env to get info
    #     # Ensure MPEEnvWrapper can be initialized with these args
    #     try:
    #         _env = MPEEnvWrapper(**mpe_env_args)
    #         _env_info = _env.get_env_info()
    #         _env.close()
    #     except Exception as e:
    #         logger.error(f"Failed to initialize temporary MPE environment with args: {mpe_env_args}")
    #         logger.error(f"Error: {e}")
    #         raise

    #     task_n_agents = _env_info["n_agents"]
    #     task_n_actions = _env_info["n_actions"]
    #     task_obs_shape = _env_info["obs_shape"]
        
    #     # Skill dimension for MPE
    #     # If skills are basic actions, skill_dim = n_actions. Otherwise, use configured skill_dim.
    #     mpe_skill_dim = task_n_actions if getattr(main_args, "mpe_basic_action_as_skill", True) else main_args.skill_dim
    #     if getattr(main_args, "mpe_basic_action_as_skill", True):
    #         logger.info(f"MPE using basic actions as skills. Skill dim: {mpe_skill_dim}")

    #     # Scheme for MPE
    #     # This needs to match what MPEEnvWrapper provides in get_obs, get_state, etc.
    #     # And how EpisodeBatch expects it.
    #     mpe_scheme = {
    #         "state": {"vshape": _env_info["state_shape"]},
    #         "obs": {"vshape": task_obs_shape, "group": "agents"},
    #         "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
    #         "avail_actions": {"vshape": (task_n_actions,), "group": "agents", "dtype": th.int},
    #         "skills": {"vshape": (1,), "group": "agents", "dtype": th.long}, # Skill index
    #         "avail_skills": {"vshape": (mpe_skill_dim,), "group": "agents", "dtype": th.int},
    #         "reward": {"vshape": (1,)}, # Global reward
    #         "terminated": {"vshape": (1,), "dtype": th.uint8},
    #     }
    #     # Add one-hot schemes if used
    #     mpe_scheme["actions_onehot"] = {"vshape": (task_n_actions,), "group": "agents", "dtype": th.float32}
    #     mpe_scheme["skills_onehot"] = {"vshape": (mpe_skill_dim,), "group": "agents", "dtype": th.float32}


    #     groups = {"agents": task_n_agents}
    #     preprocess = {
    #         "actions": ("actions_onehot", [OneHot(out_dim=task_n_actions)]),
    #         "skills": ("skills_onehot", [OneHot(out_dim=mpe_skill_dim)])
    #     }

    #     # MAC input shape for MPE (if a MAC is used beyond simple action selection)
    #     # This depends on the specific MAC architecture.
    #     # Example: if MAC uses obs, last_action, agent_id
    #     mac_input_shape = task_obs_shape 
    #     if getattr(main_args, "mac_use_last_action", False): # Example arg
    #         mac_input_shape += task_n_actions
    #     if getattr(main_args, "mac_use_agent_id", False): # Example arg
    #         mac_input_shape += task_n_agents
            
    #     task_key = f"mpe_{mpe_scenario_name}"
    #     main_args.task_info_dict[task_key] = {
    #         "env_name": "mpe", # Generic MPE identifier
    #         "runner_name": getattr(main_args, "mpe_runner_name", "mpe_hier_mcts_parallel"), # Specific MPE runner
    #         "mac_name": getattr(main_args, "mpe_mac_name", "basic_mac"), # Specific MAC for MPE
    #         "agent_name": getattr(main_args, "mpe_agent_name", "rnn"), # Agent type for MAC
    #         "env_args": mpe_env_args,
    #         "n_agents": task_n_agents,
    #         "n_actions": task_n_actions,
    #         "skill_dim": mpe_skill_dim, # Store the determined skill_dim
    #         "obs_shape": task_obs_shape,
    #         "state_shape": _env_info["state_shape"],
    #         "episode_limit": _env_info["episode_limit"],
    #         "scheme": mpe_scheme,
    #         "groups": groups,
    #         "preprocess": preprocess,
    #         "mac_input_shape": mac_input_shape,
    #         "basic_action_as_skill": getattr(main_args, "mpe_basic_action_as_skill", True)
    #     }
    #     logger.info(f"Initialized MPE task: {task_key} with n_agents={task_n_agents}, n_actions={task_n_actions}, skill_dim={mpe_skill_dim}")
    #     logger.info(f"MPE Env Args: {mpe_env_args}")
    #     logger.info(f"MPE Scheme: {mpe_scheme}")

    return (
        task2args,
        task2runner,
        task2buffer,
        task2scheme,
        task2groups,
        task2preprocess,
    )


def train_sequential(
    train_tasks,
    main_args,
    logger,
    learner,
    task2args,
    task2runner, # This runner is only for test, so it uses online interaction
    task2offlinedata,
    t_start=0,
    pretrain=False,
    test_task2offlinedata=None,
):
    ########## start training ##########
    t_env = t_start
    episode = 0  # episode does not matter
    t_max = main_args.t_max if not pretrain else main_args.pretrain_steps
    model_save_time = 0
    last_test_T = 0
    last_log_T = 0
    start_time = time.time()
    last_time = start_time
    test_time_total = 0
    test_start_time = 0

    # get some common information
    batch_size_train = main_args.batch_size
    batch_size_run = main_args.batch_size_run

    # do test before training
    n_test_runs = max(1, main_args.test_nepisode // batch_size_run)
    if main_args.evaluate:
        n_test_runs = 0
    if main_args.debug:
        n_test_runs = 0
    # test_start_time = time.time()

    # with th.no_grad():
    #     for task in main_args.test_tasks:
    #         task2runner[task].t_env = t_env
    #         # 不知道这里再搞一个test干什么，可能是为了看最开始random的性能为多少？
    #         for _ in range(n_test_runs):
    #             task2runner[task].run(test_mode=True, pretrain=pretrain)

    #     # test_pretrain for pretrained tasks
    #     if pretrain and test_task2offlinedata is not None:
    #         for task, data_buffer in test_task2offlinedata.items():
    #             episode_sample = data_buffer.sample(batch_size_train * 3)

    #             if episode_sample.device != task2args[task].device:
    #                 episode_sample.to(task2args[task].device)

    #             if hasattr(learner, "test_pretrain"):
    #                 learner.test_pretrain(episode_sample, t_env, episode, task)
    #             else:
    #                 raise ValueError(
    #                     "Do test_pretrain with a learner that does not have a `test_pretrain` method!"
    #                 )

    # test_time_total += time.time() - test_start_time
    
    # 这里每一次训练就是一次，t_max其实就是训练了t_max次
    while t_env < t_max:
        # shuffle tasks
        np.random.shuffle(train_tasks)
        # train each task
        for task in train_tasks:

            episode_sample = task2offlinedata[task].sample(batch_size_train)

            if episode_sample.device != task2args[task].device:
                episode_sample.to(task2args[task].device)

            if pretrain:
                if hasattr(learner, "pretrain"):
                    terminated = learner.pretrain(episode_sample, t_env, episode, task)
                else:
                    raise ValueError(
                        "Do pretraining with a learner that does not have a `pretrain` method!"
                    )
            else:
                terminated = learner.train(episode_sample, t_env, episode, task)

            if terminated is not None and terminated:
                break
            # 也就是说,learner的train需要有t_max次，在本文的MT setting里是21000
            t_env += 1
            episode += batch_size_run

        learner.update(pretrain=pretrain)

        if terminated is not None and terminated:
            logger.console_logger.info(
                f"Terminate training by the learner at t_env = {t_env}. Finish training."
            )
            break

        # Execute test runs once in a while & final evaluation
        if (t_env - last_test_T) / main_args.test_interval >= 1 or t_env >= t_max:
            test_start_time = time.time()

            with th.no_grad():
                # TODO: 暂时将stage 1的test去掉了，这里后面应该记录一下嘛？
                # for task in main_args.test_tasks:
                #     task2runner[task].t_env = t_env
                #     for _ in range(n_test_runs):
                #         task2runner[task].run(test_mode=True, pretrain=pretrain)

                # test_pretrain for pretrained tasks
                if pretrain and test_task2offlinedata is not None:
                    for task, data_buffer in test_task2offlinedata.items():
                        episode_sample = data_buffer.sample(batch_size_train * 10)

                        if episode_sample.device != task2args[task].device:
                            episode_sample.to(task2args[task].device)

                        if hasattr(learner, "test_pretrain"):
                            learner.test_pretrain(episode_sample, t_env, episode, task)
                        else:
                            raise ValueError(
                                "Do test_pretrain with a learner that does not have a `test_pretrain` method!"
                            )

            test_time_total += time.time() - test_start_time

            logger.console_logger.info("Step: {} / {}".format(t_env, t_max))
            logger.console_logger.info(
                "Estimated time left for stage 1 : {}. Time passed: {}. Test time cost: {}".format(
                    time_left(last_time, last_test_T, t_env, t_max),
                    time_str(time.time() - start_time),
                    time_str(test_time_total),
                )
            )
            last_time = time.time()
            last_test_T = t_env

        if main_args.save_model and (
            t_env - model_save_time >= main_args.save_model_interval
            or model_save_time == 0
        ):
            if pretrain:
                save_path = os.path.join(main_args.pretrain_save_dir, str(t_env))
            else:
                save_path = os.path.join(main_args.save_dir, str(t_env))
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))
            learner.save_models(save_path)
            model_save_time = t_env

        if (t_env - last_log_T) >= main_args.log_interval:
            last_log_T = t_env
            logger.log_stat("episode", episode, t_env)
            logger.print_recent_stats()


def train_online_mcts(
    test_tasks,
    main_args,
    logger,
    mac,
    learner,
    task2args,
    task2runner,
    task2scheme,
    task2groups,
    task2preprocess,
    task2buffer,
    t_start=0
):
    """在线MCTS训练阶段"""
    # 初始化时间相关变量
    model_save_time = 0
    last_test_T = t_start
    last_log_T = t_start
    last_target_update_T = t_start
    start_time = time.time()
    last_time = start_time
    test_time_total = 0
    
    # 初始化MCTS网络和经验回放缓冲区
    mcts_network = None
    replay_buffer_list = ReplayBufferList(capacity=main_args.replay_buffer_list_capacity if hasattr(main_args, "replay_buffer_list_capacity") else 100, use_real_data=True)

    # 初始化使用优先级采样的buffer
    use_priority_sampling = main_args.use_priority_sampling if hasattr(main_args, "use_priority_sampling") else False
    
    # 创建用于learner训练的Buffer
    if use_priority_sampling:
        # 使用优先级采样缓冲区 - 直接传入task2buffer
        priority_buffer = PrioritizedMultiTaskReplayBuffer(
            task_list=test_tasks,
            task2buffer=task2buffer,
            batch_size=main_args.batch_size,
            alpha=0.6
        )
    
    # 创建用于learner训练的MultiTaskReplayBuffer
    # 获取buffer配置参数
    
    mcts_target_update_interval = main_args.mcts_target_update_interval if hasattr(main_args, "mcts_target_update_interval") else 80
    
    # 获取一些常用参数
    batch_size_run = main_args.batch_size
    batch_size_train = main_args.batch_size

    # 设置测试参数 - 使用online专用的测试参数
    test_nepisode_online = main_args.test_nepisode_online if hasattr(main_args, "test_nepisode_online") else main_args.test_nepisode
    test_interval_online = main_args.test_interval_online if hasattr(main_args, "test_interval_online") else main_args.test_interval
    
    n_test_runs = max(1, test_nepisode_online // batch_size_run)
    if main_args.evaluate:
        n_test_runs = 0
    if main_args.debug:
        n_test_runs = 0

    # 初始化MCTS runner
    mcts_task2runner = {}
    for task in test_tasks:
        # Use the runner specified in main_args.mcts_runner
        # This should be "mpe_hier_mcts_parallel" for MPE tasks, configured in the experiment yaml/json
        current_mcts_runner_key = main_args.mcts_runner
        if task2args[task].env == "mpe" and hasattr(main_args, "mpe_mcts_runner"):
            current_mcts_runner_key = main_args.mpe_mcts_runner # Allow specific runner for MPE via config
        
        mcts_task2runner[task] = r_REGISTRY[current_mcts_runner_key](
            args=task2args[task],
            logger=logger,
            task=task
        )
        # 设置runner的初始t_env
        mcts_task2runner[task].t_env = t_start
                # 如果还没有初始化MCTS网络，现在初始化它
        total_agents = mac.get_total_agents(task)
        if mcts_network is None:
            # 从环境信息中获取观察和状态维度
            env_info = mcts_task2runner[task].get_env_info()
            state_shape = env_info["state_shape"]
            obs_shape = env_info["obs_shape"]
            # note : n_actions在这里是指技能的数量
            # For MPE, n_actions should be the discrete action space size from the wrapper
            n_actions = env_info["n_actions"] 
            if getattr(main_args,"basic_action_as_skill", False) or task2args[task].env == "mpe":
                # 如果使用基本动作作为技能，则将技能维度设置为动作维度
                main_args.skill_dim = n_actions
                task2args[task].skill_dim = n_actions # Ensure task_args is also updated for consistency
            
            n_agents = env_info["n_agents"]
            input_shape = obs_shape # This is individual agent observation
            
            # The PolicyRNN in mctx example might expect concatenated agent obs or global state
            # For MPE, individual agent obs is common. If PolicyRNN needs global state for embedding,
            # ensure emb_input_shape is set correctly.
            # The current PolicyRNN takes obs_input_shape for individual agent policy
            # and emb_input_shape for value function (often global state).

            if task2args[task].obs_last_action:
                # This is for PolicyRNN input, skill_dim is used as it represents the action space for MCTS
                input_shape += task2args[task].skill_dim 
            if task2args[task].obs_agent_id:
                input_shape += n_agents
                
            # 初始化PolicyRNN网络
            mcts_network = PolicyRNN(
                obs_input_shape=input_shape, # Per-agent observation shape
                emb_input_shape=state_shape, # Global state shape for value embedding
                # 对于gumbel来说，output shape和n_action是一样的 (skill_dim)
                output_shape=task2args[task].skill_dim, # Output is skill/action probabilities
                num_agents=n_agents, # Number of agents
                c_step=mac.c_step,
                device=main_args.device,
                optimizer='adam',
                hidden_dim=main_args.hidden_dim if hasattr(main_args, "hidden_dim") else 128,
                seed=main_args.seed if hasattr(main_args, "seed") else 42
            )
            mcts_network.init_hidden(batch_size=batch_size_run)
        
        if main_args.mcts_network_path != "":
            # 加载预训练的MCTS网络参数
            mcts_network.load_state_dict(th.load(os.path.join(main_args.mcts_network_path, "mcts_network.th"), map_location=main_args.device))
            mcts_network.update_target_network()
            logger.console_logger.info(f"Loaded MCTS network from {main_args.mcts_network_path}")

        # 设置runner，传入learner的MAC作为基础策略
        mcts_task2runner[task].setup(
            scheme=task2scheme[task],
            groups=task2groups[task],
            preprocess=task2preprocess[task],
            mac=mac,
            mcts_network=mcts_network
        )

    logger.console_logger.info("Beginning online MCTS training stage...")
    
    # 主训练循环
    current_t_env = t_start
    episode = 0  # 仅用于learner的episode计数
    train_count = 0
    last_update_priorities = {}  # 记录每个任务上次更新优先级的时间
    
    # 初始化wandb使用标志
    use_wandb = main_args.use_wandb if hasattr(main_args, "use_wandb") else False
    
    # 添加训练统计跟踪
    mcts_train_stats = {
        "loss": [],
        "policy_loss": [],
        "value_loss": [],
    }
    
    # 添加测试统计跟踪
    test_stats = {}
    for task in test_tasks:
        test_stats[task] = {"return": [], "win_rates": []}

    # 添加训练任务统计
    train_task_stats = {}
    for task in test_tasks:
        train_task_stats[task] = {"returns": [], "win_rates": [], "lengths": []}

    # 添加窗口平均统计
    window_size = 20  # 使用相同大小的窗口计算移动平均
    train_windows = {}
    for task in test_tasks:
        train_windows[task] = {"returns": [], "win_rates": []}

    while current_t_env < main_args.online_steps:
        # 收集在线数据
        for task in test_tasks:
            # 使用MCTS收集经验
            episode_data = mcts_task2runner[task].run(test_mode=False)
            episode_batch, mcts_buffer, stats_info = episode_data
            
            # 将收集到的episode_batch存入buffer
            if use_priority_sampling:
                priority_buffer.insert_episode_batch(task, episode_batch)
            else:
                task2buffer[task].insert_episode_batch(episode_batch)
            
            # 只有当buffer中有足够的数据时才进行learner训练
            sample_batch_size = main_args.sample_batch_size if hasattr(main_args, "sample_batch_size") else 32
            
            if use_priority_sampling:
                if priority_buffer.can_sample(task, sample_batch_size):
                    # 使用优先级采样，简化为不需要weights
                    training_batch, indices = priority_buffer.sample(task, sample_batch_size)
                    
                    # 确保数据在正确的设备上
                    if training_batch.device != main_args.device:
                        training_batch.to(main_args.device)
                    
                    # 计算TD误差用于更新优先级
                    losses = learner.train(training_batch, current_t_env, episode, task, use_external_skill=True, return_losses=True)
                    
                    # 更新样本优先级
                    if losses is not None and isinstance(losses, dict):
                        value_loss = losses.get('value_loss', 0.0)
                        reward_loss = losses.get('reward_loss', 0.0)
                        # 组合损失作为优先级
                        priorities = value_loss + reward_loss + 1e-6  # 添加小值避免零优先级
                        priority_buffer.update_priorities(task, indices, priorities)
            else:
                if task2buffer[task].can_sample(sample_batch_size):
                    training_batch = task2buffer[task].sample(sample_batch_size)
                    
                    # 确保数据在正确的设备上
                    if training_batch.device != main_args.device:
                        training_batch.to(main_args.device)
                    
                    # 使用external_skill=True表示我们使用MCTS生成的技能进行训练
                    learner.train(training_batch, current_t_env, episode, task, use_external_skill=True)
            
            # 将收集到的MCTS经验添加到replay_buffer_list
            if mcts_buffer is not None:
                replay_buffer_list.push(mcts_buffer)
            
            # 更新当前的环境步数
            current_t_env = mcts_task2runner[task].t_env
            
            episode += batch_size_run
            
            # 记录训练任务的统计信息
            episode_return = stats_info["episode_return"]
            win = stats_info["win_rate"]
            episode_length = stats_info["episode_length"]
            
            # 添加到训练统计
            # Note: not use？
            train_task_stats[task]["returns"].append(episode_return)
            train_task_stats[task]["lengths"].append(episode_length)
            if win is not None:
                train_task_stats[task]["win_rates"].append(win)
                
            # 更新移动窗口数据
            train_windows[task]["returns"].append(episode_return)
            if len(train_windows[task]["returns"]) > window_size:
                train_windows[task]["returns"].pop(0)
            
            if win is not None:
                train_windows[task]["win_rates"].append(win)
                if len(train_windows[task]["win_rates"]) > window_size:
                    train_windows[task]["win_rates"].pop(0)
            
            # 记录到wandb
            if use_wandb:
                # 计算移动平均
                window_return_avg = np.mean(train_windows[task]["returns"]) if train_windows[task]["returns"] else 0
                window_win_avg = np.mean(train_windows[task]["win_rates"]) if train_windows[task]["win_rates"] else 0
                
                task_stats = {
                    f"train/{task}/episode_return": episode_return,
                    f"train/{task}/episode_length": episode_length,
                    f"train/{task}/window_return_avg": window_return_avg
                }
                
                if win is not None:
                    task_stats[f"train/{task}/win_rates"] = win
                    task_stats[f"train/{task}/window_win_avg"] = window_win_avg
                
                # 记录其他环境信息
                for k, v in stats_info.get("stats", {}).items():
                    if isinstance(v, (int, float)):
                        task_stats[f"train/{task}/{k}"] = v
                
                wandb.log(task_stats, step=current_t_env)
        
        # 训练MCTS网络
        train_start = time.time()
        # TODO: just for test
        # if len(replay_buffer_list) >= batch_size_train and current_t_env >= 3000:
        if len(replay_buffer_list) >= batch_size_train:
            
            # 从ReplayBufferList中采样数据
            batch = replay_buffer_list.sample(batch_size_train)
            
            # 训练MCTS网络
            loss_dict = mcts_network.train_network(
                batch=batch,
                gamma=main_args.gamma if hasattr(main_args, "gamma") else 0.99,
                value_loss_weight=main_args.value_loss_weight if hasattr(main_args, "value_loss_weight") else 0.5,
                max_grad_norm=main_args.max_grad_norm if hasattr(main_args, "max_grad_norm") else 10.0,
            # 在这个代码里应该一直是使用real_data 进行训练的
                # TODO: just for test
                use_real_data=False,
            )
            
            # 记录训练统计信息
            loss = loss_dict.get("total_loss", 0.0)
            policy_loss = loss_dict.get("policy_loss", 0.0)
            value_loss = loss_dict.get("value_loss", 0.0)
            entropy = loss_dict.get("entropy", 0.0)
            logger.log_stat("loss", loss, current_t_env)
            logger.log_stat("policy_loss", policy_loss, current_t_env)
            logger.log_stat("value_loss", value_loss, current_t_env)
            logger.log_stat("entropy", entropy, current_t_env)
            # mcts_train_stats["loss"].append(loss)
            # mcts_train_stats["policy_loss"].append(policy_loss)
            # mcts_train_stats["value_loss"].append(value_loss)
            
            # 使用wandb记录训练信息
            if use_wandb:
                wandb.log({
                    "mcts/total_loss": loss,
                    "mcts/policy_loss": policy_loss,
                    "mcts/value_loss": value_loss,
                    "mcts/entropy": entropy,
                    # "mcts/buffer_size": len(replay_buffer_list),
                    # "mcts/train_time": time.time() - train_start
                }, step=current_t_env)
            
            train_count += 1
            
            # 定期更新目标网络
            if (episode - last_target_update_T) / mcts_target_update_interval >= 1:
                last_target_update_T = episode
                mcts_network.update_target_network()
                logger.console_logger.info(f"Updated MCTS network target at t_env: {current_t_env}, episode: {episode}")
                
            # 定期记录平均损失，记录平均损失后清空mcts_train_stats。这个是只有可以训练的时候才记录，还不能和下面的logger记录相合并
            # if (current_t_env - last_log_T) >= main_args.log_interval:
            #     if use_wandb:
            #         # 记录平均损失
            #         wandb.log({
            #             "mcts/avg_loss": np.mean(mcts_train_stats["loss"]),
            #             "mcts/avg_policy_loss": np.mean(mcts_train_stats["policy_loss"]),
            #             "mcts/avg_value_loss": np.mean(mcts_train_stats["value_loss"]),
            #         }, step=current_t_env)
                
            #     # 重置统计信息
            #     for k in mcts_train_stats:
            #         mcts_train_stats[k] = []
        
        # 定期测试 - 使用online专用的测试间隔
        if (current_t_env - last_test_T) / test_interval_online >= 1 or current_t_env >= main_args.online_steps:
            test_start_time = time.time()
            # TODO: just for test, do not test now
            # # 清空本轮测试统计数据
            # for task in test_tasks:
            #     test_stats[task]["returns"] = []
            #     test_stats[task]["win_rates"] = []
            #     test_stats[task]["lengths"] = []
            
            # logger.console_logger.info(f"------- Testing at t_env={current_t_env} -------")
            
            # with th.no_grad():
            #     for task in test_tasks:
            #         mcts_task2runner[task].t_env = current_t_env
            #         # 执行测试episode
            #         for _ in range(n_test_runs):
            #             episode_data = mcts_task2runner[task].run(test_mode=True)
            #             _, _, stats_info = episode_data
                        
            #             # 收集测试统计信息
            #             test_stats[task]["returns"].append(stats_info["episode_return"])
            #             test_stats[task]["lengths"].append(stats_info["episode_length"])
                        
            #             if stats_info["win_rate"] is not None:
            #                 test_stats[task]["win_rates"].append(stats_info["win_rate"])
            
            # # 记录测试统计到控制台 - 只显示测试结果
            # logger.console_logger.info("Test Results Summary:")
            # for task in test_tasks:
            #     task_mean_test_return = np.mean(test_stats[task]["returns"]) if test_stats[task]["returns"] else 0
            #     task_std_test_return = np.std(test_stats[task]["returns"]) if test_stats[task]["returns"] else 0
                
            #     win_msg = ""
            #     if test_stats[task]["win_rates"]:
            #         test_win_rate = np.mean(test_stats[task]["win_rates"])
            #         win_msg = f", Win Rate: {test_win_rate:.3f}"
                
            #     logger.console_logger.info(
            #         f"Task {task} - Return: {task_mean_test_return:.3f} ± {task_std_test_return:.3f}{win_msg}"
            #     )
            
            # # 记录测试统计到wandb - 只记录测试数据
            # if use_wandb:
            #     # all_test_returns = []
            #     # all_test_wins = []
                
            #     # 只准备测试数据
            #     test_data = {}
                
            #     for task in test_tasks:
            #         # 计算测试统计均值
            #         task_mean_test_return = np.mean(test_stats[task]["returns"]) if test_stats[task]["returns"] else 0
            #         task_std_test_return = np.std(test_stats[task]["returns"]) if test_stats[task]["returns"] else 0
                   
            #         # 记录测试数据
            #         test_data[f"test/{task}/mean_return"] = task_mean_test_return
            #         test_data[f"test/{task}/std_return"] = task_std_test_return
            #         test_data[f"test/{task}/mean_length"] = np.mean(test_stats[task]["lengths"]) if test_stats[task]["lengths"] else 0
                   
            #         # 如果有win_rate统计，也记录它
            #         if test_stats[task]["win_rates"]:
            #             test_win_rate = np.mean(test_stats[task]["win_rates"])
            #             test_data[f"test/{task}/win_rates"] = test_win_rate

                
            #     # 记录测试时间
            #     test_data["test/test_time"] = time.time() - test_start_time
                
            #     # 记录测试数据
            #     wandb.log(test_data, step=current_t_env)
                
            #     # 添加测试评估标记点
            #     wandb.log({"test/evaluation": current_t_env}, step=current_t_env)
            
            # # 记录测试统计到日志
            # for task in test_tasks:
            #     task_mean_return = np.mean(test_stats[task]["returns"]) if test_stats[task]["returns"] else 0
            #     logger.log_stat(f"test_return_mean_{task}", task_mean_return, current_t_env)
                
            #     if test_stats[task]["win_rates"]:
            #         win_rate = np.mean(test_stats[task]["win_rates"])
            #         logger.log_stat(f"test_win_rate_{task}", win_rate, current_t_env)
            
            test_time_total += time.time() - test_start_time
            
            logger.console_logger.info("Online Step: {} / {}".format(current_t_env, main_args.online_steps))
            logger.console_logger.info(
                "Estimated time left for state 2 : {}. Time passed: {}. Test time cost: {}".format(
                    time_left(last_time, last_test_T, current_t_env, main_args.online_steps),
                    time_str(time.time() - start_time),
                    time_str(test_time_total),
                )
            )
            logger.console_logger.info("--------------------------------------")
            
            last_time = time.time()
            last_test_T = current_t_env
        
        # 定期保存模型
        if main_args.save_model and (
            current_t_env - model_save_time >= main_args.save_model_interval_online
            or model_save_time == 0
        ):
            save_path = os.path.join(main_args.online_save_dir, str(current_t_env))
            os.makedirs(save_path, exist_ok=True)            
            # 保存MCTS网络模型
            mcts_net_path = os.path.join(save_path, "mcts_network.th")
            th.save(mcts_network.state_dict(), mcts_net_path)
            logger.console_logger.info(f"Saved MCTS network to {mcts_net_path}")

            logger.console_logger.info("Saving models to {}".format(save_path))
            learner.save_models(save_path)
            
            model_save_time = current_t_env
        
        # 定期记录日志
        if (current_t_env - last_log_T) >= main_args.log_interval:
            last_log_T = current_t_env   
            logger.log_stat("episode", episode, current_t_env)
            logger.print_recent_stats()
    
    # 保存最终模型
    if main_args.save_model:
        save_path = os.path.join(main_args.online_save_dir, str(current_t_env))
        os.makedirs(save_path, exist_ok=True)
        # 保存MCTS网络模型
        mcts_net_path = os.path.join(save_path, "mcts_network.th")
        th.save(mcts_network.state_dict(), mcts_net_path)
        logger.console_logger.info(f"Saved final MCTS network to {mcts_net_path}")
    
    logger.console_logger.info("Finished online MCTS training")


def run_sequential(args, logger):
    # Init runner so we can get env info
    args.n_tasks = len(args.train_tasks)
    # define main_args
    main_args = copy.deepcopy(args)
    if getattr(main_args, "pretrain", False):
        all_tasks = list(set(args.train_tasks + args.test_tasks + args.pretrain_tasks))
    else:
        all_tasks = list(set(args.train_tasks + args.test_tasks))

    task2args, task2runner, task2buffer, task2scheme, task2groups, task2preprocess = (
        init_tasks(all_tasks, main_args, logger)
    )
    task2buffer_scheme = {task: task2buffer[task].scheme for task in all_tasks}
    if getattr(main_args, "basic_action_as_skill", False):
        # single task, 只有一个任务
        for task in all_tasks:
            n_actions = task2args[task].n_actions
            break
        main_args.skill_dim = n_actions
    # define mac
    mac = mac_REGISTRY[main_args.mac](
        train_tasks=all_tasks,
        task2scheme=task2buffer_scheme,
        task2args=task2args,
        main_args=main_args,
    )

    for task in main_args.test_tasks:
        task2runner[task].setup(
            scheme=task2scheme[task],
            groups=task2groups[task],
            preprocess=task2preprocess[task],
            mac=mac,
        )

    # define learner
    learner = le_REGISTRY[main_args.learner](mac, logger, main_args)

    if main_args.use_cuda:
        learner.cuda()

    if main_args.checkpoint_path != "":
        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(main_args.checkpoint_path):
            logger.console_logger.info(
                "Checkpoint directiory {} doesn't exist".format(
                    main_args.checkpoint_path
                )
            )
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(main_args.checkpoint_path):
            full_name = os.path.join(main_args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if main_args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(
                timesteps, key=lambda x: abs(x - main_args.load_step)
            )

        model_path = os.path.join(main_args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)

        if main_args.evaluate or main_args.save_replay:
            evaluate_sequential(main_args, logger, task2runner)
            return
    # 我们的代码中没有可以加载的pretrain learner，直接学就行了,不需要用其他任务先pretrain一下
    if (
        getattr(main_args, "pretrain", True)
        and getattr(main_args, "agent") != "mt_odis_ns"
    ):
        # initialize training data for each task
        task2offlinedata = {}
        for task in main_args.pretrain_tasks:
            # create offline data buffer
            task2offlinedata[task] = OfflineBuffer(
                task,
                main_args.pretrain_tasks_data_quality[task],
                data_folder=main_args.offline_data_name,
                offline_data_size=args.offline_data_size,
                random_sample=args.offline_data_shuffle,
            )

        test_task2offlinedata = None
        # add test data if learner has `test_pretrain` function
        if hasattr(learner, "test_pretrain") and hasattr(
            main_args, "test_tasks_data_quality"
        ):
            test_task2offlinedata = {}
            for task in main_args.test_tasks_data_quality.keys():
                test_task2offlinedata[task] = OfflineBuffer(
                    task,
                    main_args.test_tasks_data_quality[task],
                    data_folder=main_args.offline_data_name,
                    offline_data_size=args.offline_data_size,
                    random_sample=args.offline_data_shuffle,
                )

        logger.console_logger.info(
            "Beginning pre-training with {} timesteps for each task".format(
                main_args.pretrain_steps
            )
        )
        train_sequential(
            main_args.pretrain_tasks,
            main_args,
            logger,
            learner,
            task2args,
            task2runner,
            task2offlinedata,
            pretrain=True,
            test_task2offlinedata=test_task2offlinedata,
        )
        logger.console_logger.info(f"Finished pretraining")
        test_task2offlinedata = None  # free memory

        save_path = os.path.join(
            main_args.pretrain_save_dir, str(main_args.pretrain_steps)
        )
        os.makedirs(save_path, exist_ok=True)
        logger.console_logger.info("Saving models to {}".format(save_path))
        learner.save_models(save_path)

    # initialize training data for each task
    task2offlinedata = {}
    for task in main_args.train_tasks:
        # create offline data buffer
        task2offlinedata[task] = OfflineBuffer(
            task,
            main_args.train_tasks_data_quality[task],
            data_folder=main_args.offline_data_name,
            dataset_folder=args.offline_data_folder,
            offline_data_size=args.offline_data_size,
            random_sample=args.offline_data_shuffle,
        )

    logger.console_logger.info(
        "Beginning multi-task offline training with {} timesteps for each task".format(
            main_args.t_max
        )
    )
    if main_args.load_wm_path != "":
        learner.load_models(main_args.load_wm_path)
        if getattr(main_args,"test_offline_model", False):
            # test the offline model
            for task in main_args.test_tasks:
                task2runner[task].t_env = 0
                for _ in range(main_args.test_nepisode):
                    task2runner[task].run(test_mode=True, pretrain=False)
            return
    # Stage 1 : train each task with offline data
    if getattr(main_args, "use_offline_training", False):
        train_sequential(
            main_args.train_tasks,
            main_args,
            logger,
            learner,
            task2args,
            task2runner,
            task2offlinedata,
        )

        # save the final model
        if main_args.save_model:
            save_path = os.path.join(main_args.save_dir, str(main_args.t_max))
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving final models to {}".format(save_path))
            learner.save_models(save_path)

    # Stage 2 : online training with hierarchical ma gumbel muzero
    if getattr(main_args, "use_online_mcts", False):
        # 创建在线训练的保存目录
        if not hasattr(main_args, "online_save_dir"):
            main_args.online_save_dir = os.path.join(main_args.results_save_dir, "online_models")
        os.makedirs(main_args.online_save_dir, exist_ok=True)
        
        # 设置在线学习的时长
        if not hasattr(main_args, "online_steps"):
            main_args.online_steps = main_args.t_max  # 默认与离线训练相同
        # 启动在线MCTS训练
        train_online_mcts(
            main_args.test_tasks,
            main_args,
            logger,
            mac,
            learner,
            task2args,
            task2runner,
            task2scheme,
            task2groups,
            task2preprocess,
            task2buffer,
            t_start=0
        )
    else:
        # 如果不使用在线MCTS，关闭环境
        for task in args.test_tasks:
            task2runner[task].close_env()
    
    logger.console_logger.info(f"Finished Training")


def args_sanity_check(config, _log):
    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning(
            "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!"
        )

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (
            config["test_nepisode"] // config["batch_size_run"]
        ) * config["batch_size_run"]

    return config
