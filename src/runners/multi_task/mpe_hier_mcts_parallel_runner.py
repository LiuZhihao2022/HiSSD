import sys
import logging
import functools # Added for partial

# Adjust paths to import necessary modules from HiSSD and OMAR-master
# sys.path.append('/home/lzh/HiSSD') # Already added in calling scripts usually
# sys.path.append('/home/lzh/HiSSD/OMAR-master') # If OMAR components are needed directly

# Assuming these are standard imports from the project
from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe, Process
import numpy as np # Added for env_worker example
import cloudpickle # Added for CloudpickleWrapper

# FROM policy_improvement_demo.py (mctx related)
# Ensure these paths are correct or mctx is installed
# sys.path.append('/home/lzh/HiSSD') # Redundant if already in sys.path
from typing import Tuple, Optional
from absl import app # If used
from absl import flags # If used
import jax # If mctx uses jax
import mctx # Main mctx import
import copy
# Assuming these are part of mctx or examples
from mctx._src.network import PolicyRNN, ReplayBuffer, compute_prior_from_qvalues # Example, adjust if needed
from mctx._src.optimizer_wrapper import ValueOptimizerWrapper # Example, adjust if needed
# from mctx._src.simple_env import SimpleEnv # Example, adjust if needed
from mctx._src.recurrent_fn import make_recurrent_fn_gym, make_recurrent_fn_world_model, make_multiagent_recurrent_fn_gym # Example, adjust if needed
from mctx._src.utils import stochastic_top_k_sampling # Example, adjust if needed
from mctx._src import action_selection # Example, adjust if needed
# from examples.policy_improvement_demo import initialize_root, DemoOutput # If these specific items are used

# JAX and logging configurations (copied from HierMCTSParallelRunner)
if 'jax' in sys.modules: # Check if jax is imported before configuring
    jax.config.update('jax_debug_nans', True)
logging.getLogger('jax').setLevel(logging.INFO)
logging.getLogger('absl').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.ERROR)


class CloudpickleWrapper():
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        self.x = cloudpickle.loads(ob)
    def __call__(self, *args, **kwargs): # Make it callable if x is a function
        return self.x(*args, **kwargs)


def env_worker(remote, env_fn_wrapper):
    """
    Worker process for handling an environment instance.
    """
    env = env_fn_wrapper.x()  # Unwrap the cloudpickled environment factory and call it
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "get_env_info":
                remote.send(env.get_env_info())
            elif cmd == "reset":
                env.reset()
                # For MPE, get_obs() might return a dict of agent observations
                # get_state() might be concatenation or not available.
                # This needs to align with how EpisodeBatch is populated.
                remote.send({"obs": env.get_obs(), "state": env.get_state() if hasattr(env, 'get_state') else None})
            elif cmd == "step":
                actions = data
                # MPE step returns: rewards, dones, infos (usually dicts per agent)
                # result = env.step(actions) # This is too generic
                # We need to unpack based on MPE's typical return (often PettingZoo like)
                # For now, assuming env.step returns what's needed by the runner's batch processor
                # This part is highly dependent on the MPE wrapper's API
                
                # Example for a PettingZoo-like MPE env:
                # rewards_dict, dones_dict, infos_dict = {}, {}, {}
                # agent_obs_dict = {}
                # global_reward = 0
                # global_done = False
                # for agent_id in env.agents: # Assuming PettingZoo API
                #    if not env.dones[agent_id]:
                #        obs, reward, done, truncated, info = env.last() # Get last observation for agent
                #        env.step(actions[agent_id] if isinstance(actions, dict) else actions) # Step for current agent
                #        agent_obs_dict[agent_id] = obs
                #        rewards_dict[agent_id] = reward
                #        dones_dict[agent_id] = done or truncated
                #        infos_dict[agent_id] = info
                #        global_reward += reward # Summing rewards as an example
                #        if dones_dict[agent_id]: global_done = True # If any agent is done

                # This is a placeholder for actual MPE step processing logic.
                # The runner expects specific data structure after a step.
                # Let's assume the MPE wrapper's step() method returns a tuple
                # (reward, terminated, info) where reward is global or per-agent based on config,
                # and info contains agent-specific details if needed.
                # And obs/state are retrieved via get_obs()/get_state().

                reward, terminated, info = env.step(actions) # Simplified, MPE wrapper must adapt
                
                # The runner will likely expect:
                # obs, state (if available), reward (scalar or vector), terminated (scalar), info (dict)
                # This needs to be consistent with how EpisodeBatch is filled.
                remote.send({
                    "obs": env.get_obs(),
                    "state": env.get_state() if hasattr(env, 'get_state') else None,
                    "reward": reward,
                    "terminated": terminated,
                    "info": info
                })

            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_stats": # From PyMARL, might be useful
                 remote.send(env.get_stats() if hasattr(env, "get_stats") else {})
            else:
                remote.send({"error": "Unknown command"})
    except EOFError: # Pipe closed
        pass
    except Exception as e:
        logging.error(f"Error in env_worker: {e}")
        remote.send({"error": str(e)})
    finally:
        if hasattr(env, 'close'):
            env.close()


class MPEHierMCTSParallelRunner:

    def __init__(self, args, logger, task):
        self.args = args
        self.logger = logger
        self.task = task # task here is the MPE scenario name e.g. "simple_spread_v3"
        self.batch_size = self.args.batch_size_run

        # Create environment subprocesses
        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])
        # Ensure env_REGISTRY[self.args.env] correctly points to MPEEnvWrapper
        env_fn = env_REGISTRY[self.args.env] 

        worker_id2env_args = {}
        for worker_id in range(self.batch_size):
            worker_id2env_args[worker_id] = copy.deepcopy(self.args.env_args)
            # Ensure env_args contains scenario_name for MPE, and seed is handled
            if "seed" not in worker_id2env_args[worker_id]:
                 worker_id2env_args[worker_id]["seed"] = self.args.seed # Use a base seed from args
            worker_id2env_args[worker_id]["seed"] += worker_id
            # Pass the task (scenario_name) to the MPE environment if needed by its constructor
            # worker_id2env_args[worker_id]["scenario_name"] = self.task 
            # This depends on how MPEEnvWrapper and underlying MPE env are initialized.
            # Assuming env_args already contains scenario_name or key.
            
        self.ps = [Process(target=env_worker,
                           args=(worker_conn, CloudpickleWrapper(partial(env_fn, **worker_id2env_args[worker_id]))))
                   for worker_id, worker_conn in enumerate(self.worker_conns)]

        for p in self.ps:
            p.daemon = True
            p.start()

        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]
        self.n_agents = self.env_info["n_agents"] # Get n_agents from env_info

        self.t = 0
        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self.log_train_stats_t = -100000
        
        # MCTS parameters
        self.c_step = args.c_step
        self.num_simulations = getattr(args, "num_simulations", 32)
        self.max_num_considered_actions = getattr(args, "max_num_considered_actions", 16)
        self.use_mixed_value = getattr(args, "use_mixed_value", False)
        self.k = getattr(args, "k", 10) # k for top-k sampling in MCTS
        self.temperature = getattr(args, "temperature", 1.0)

        # Parallel related state variables
        self.current_skill_indices = [None for _ in range(self.batch_size)]
        self.current_mcts_data = [None for _ in range(self.batch_size)]
        self.last_skill_selection_t = [0 for _ in range(self.batch_size)]
        self.accumulated_rewards = [0 for _ in range(self.batch_size)]
        self.wm_hidden_states = None
        
        # Ensure env_args used for PRNGKey seed is consistent if MPEEnvWrapper uses it.
        # Or use a global seed from self.args.seed
        rng_seed = self.args.env_args.get('seed', self.args.seed) if hasattr(self.args, 'env_args') and self.args.env_args else self.args.seed
        self.rng_key = jax.random.PRNGKey(rng_seed)
        
        root_action_selection_fn=functools.partial(
          action_selection.gumbel_muzero_root_action_selection,
          num_simulations=self.num_simulations,
          max_num_considered_actions=self.max_num_considered_actions,
          qtransform=functools.partial(
                mctx.qtransform_completed_by_mix_value,
                use_mixed_value=self.use_mixed_value,
            ),
        )
        
        interior_action_selection_fn=functools.partial(
            action_selection.gumbel_muzero_interior_action_selection,
            qtransform=functools.partial(
                mctx.qtransform_completed_by_mix_value,
                use_mixed_value=self.use_mixed_value,
            ),
        )

        self.action_selection_fn = action_selection.switching_action_selection_wrapper(
            root_action_selection_fn=root_action_selection_fn,
            interior_action_selection_fn=interior_action_selection_fn
        )

        # MCTS distribution visualization attributes
        self.episode_count = 0
        self.distribution_record_interval = getattr(args, "distribution_record_interval", 16)
        self.distribution_records = []
        self.max_distributions_to_keep = getattr(args, "max_distributions_to_keep", 150)
        
        # Bandit test related attributes (if keeping this functionality)
        self.bandit_run_count = 0
        self.bandit_stats_history = {
            "mean_advantage": [],
            "mean_selected_action_value": [],
            "mean_prior_policy_action_value": [],
            "mean_action_weights_policy_value": [],
            "mean_root_value": [],
            "mean_q_value_advantage": [],
            "prior_action_probs": [],
            "prior_action_logits": [],
            "mcts_action_weights": []
        }


    def setup(self, scheme, groups, preprocess, mac, mcts_network):
        self.new_batch = partial(
            EpisodeBatch,
            scheme,
            groups,
            self.batch_size,
            self.episode_limit + 1,
            preprocess=preprocess,
            device=self.args.device,
        )
        self.mac = mac # MAC should be MPE-compatible
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess
        self.mcts_network = mcts_network # MCTSNetwork (PolicyRNN) should be MPE-compatible

        # self.n_agents is already set in __init__ from self.env_info
        # Removed SMAC-specific: self.task_decomposer, self.n_enemy, self.n_ally

        # Create recurrent_fn for MCTS
        # Ensure make_recurrent_fn_world_model and mcts_network (PolicyRNN) are compatible with MPE.
        # The PolicyRNN's initial_inference and recurrent_inference will be key.
        self.recurrent_fn = make_recurrent_fn_world_model(
            self.mcts_network, # Pass the network directly
            self.mac, # Not needed if mcts_network handles predictions
            self.batch_size, # Not needed by this version of make_recurrent_fn_world_model
            self.temperature,
            self.n_agents, # Not directly needed by make_recurrent_fn_world_model
            self.k, # Not directly needed by make_recurrent_fn_world_model
            # The following offline_value args are specific to a certain world model setup.
            # May need adjustment or removal if not applicable to MPE world model.
            offline_value_start=self.args.offline_value_start,
            offline_value_end=self.args.offline_value_end,
            offline_value_anneal_time=self.args.offline_value_anneal_time
        )
        
        # MCTS replay buffers for each environment
        self.replay_buffers_mcts = [ReplayBuffer(
            self.episode_limit // self.c_step + 1,
            1,  # Each environment has its own buffer, so batch_size for buffer is 1
            self.c_step,
            use_real_data=getattr(self.args, "use_real_data", False) # Configurable
        ) for _ in range(self.batch_size)]

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        # Placeholder: Implement if replay saving is needed for MPE
        self.logger.console_logger.info("Replay saving not implemented for MPE runner.")
        pass

    def close_env(self):
        for parent_conn in self.parent_conns:
            try:
                parent_conn.send(("close", None))
                parent_conn.close() # Close on runner side
            except IOError: # Already closed
                pass
        for p in self.ps:
            p.join()
        self.logger.console_logger.info(f"Closed MPE environments for task {self.task}")


    def reset(self):
        """Resets all parallel environments."""
        self.batch = self.new_batch() # Create a new batch for the episode

        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))

        # Collect reset results (initial obs, state)
        # This needs to align with how EpisodeBatch is filled at t=0
        pre_transition_data = {
            "state": [],
            "avail_actions": [], # MPE might not have avail_actions, or wrapper provides it
            "obs": []
        }
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            # Example: data = {"obs": obs_dict, "state": state_arr}
            # The MPE wrapper must provide obs and state in a format that EpisodeBatch expects
            # or this runner must transform it.
            # For MPE, obs is often a dict of agent observations.
            # state might be a global state or None.
            # avail_actions might be all actions if not restricted.
            
            # This is a simplified placeholder. Actual data processing depends on MPE wrapper and scheme.
            # Assuming env_info provides n_agents and n_actions for avail_actions shape.
            # And obs/state shapes are also in env_info.
            pre_transition_data["state"].append(data.get("state", np.zeros(self.env_info["state_shape"])))
            pre_transition_data["obs"].append(data.get("obs", np.zeros((self.env_info["n_agents"], self.env_info["obs_shape"]))))
            pre_transition_data["avail_actions"].append(
                np.ones((self.env_info["n_agents"], self.env_info["n_actions"]))
            )


        self.batch.update(pre_transition_data, ts=0)
        self.t = 0 # Reset episode time step counter
        # Reset other episode-specific states
        self.current_skill_indices = [None for _ in range(self.batch_size)]
        self.current_mcts_data = [None for _ in range(self.batch_size)]
        self.last_skill_selection_t = [0 for _ in range(self.batch_size)]
        self.accumulated_rewards = [0.0 for _ in range(self.batch_size)]


    def run(self, test_mode=False, nolog=False, pretrain=False):
        """
        Runs a single episode for each parallel environment.
        Returns episode batch, mcts_buffer (if any), and stats.
        This is a STUB based on the provided hier_mcts_parallel_runner.py.
        A full implementation would involve a loop for self.episode_limit steps.
        """
        self.reset() # Reset environments and batch
        
        all_terminated = False
        episode_returns = [0.0] * self.batch_size
        episode_lengths = [0] * self.batch_size
        env_steps_this_episode = 0

        # Placeholder for MCTS related data if needed by the training loop
        mcts_trajectory_data = None # e.g. list of (root_value, actions, policy_probs)

        # Main episode loop
        while not all_terminated and self.t < self.episode_limit:
            # TODO: Implement MCTS based action selection here for each agent/environment
            # 1. Get current observations/states from self.batch at self.t
            # 2. For each environment in the batch:
            #    a. Construct MCTS root node
            #    b. Run MCTS simulations using self.mcts_network and self.action_selection_fn
            #    c. Select action(s)
            # This is a complex part involving interaction with mctx library.
            # For now, using random actions as a placeholder.
            
            actions = [] # List of actions for each parallel environment
            for i in range(self.batch_size):
                # Placeholder: Get available actions for each agent
                # avail_actions_i = self.batch["avail_actions"][self.t, i] # Shape: (n_agents, n_actions)
                # For MPE, n_actions is likely fixed per agent.
                # Assuming self.env_info["n_actions"] is the number of discrete actions.
                # And self.env_info["n_agents"] is the number of agents.
                # This needs to be a list of action arrays/tensors, one per env in batch.
                # Each element itself could be an array of actions for n_agents.
                # Example: np.random.randint(0, self.env_info["n_actions"], size=self.env_info["n_agents"])
                # The exact structure depends on what env.step() expects.
                # If MPE env expects a dict of actions per agent, this needs to be formatted.
                # For now, assume actions is a list of np arrays, where each array is (n_agents,).
                agent_actions = np.random.randint(0, self.env_info.get("n_actions", 1), size=self.env_info.get("n_agents",1))
                actions.append(agent_actions)


            # Send actions to environments
            for i in range(self.batch_size):
                self.parent_conns[i].send(("step", actions[i]))

            # Collect results from environments
            post_transition_data = {
                "reward": [],
                "terminated": [],
                # "obs": [], # obs for next step
                # "state": [], # state for next step
                # "avail_actions": [] # avail_actions for next step
            }
            obs_list = []
            state_list = []
            avail_actions_list = []

            all_terminated_this_step = True
            for i in range(self.batch_size):
                data = self.parent_conns[i].recv() # data = {"obs":..., "state":..., "reward":..., "terminated":..., "info":...}
                
                episode_returns[i] += data["reward"] # Assuming reward is scalar or summed for the env
                post_transition_data["reward"].append(data["reward"])
                post_transition_data["terminated"].append(data["terminated"])
                
                obs_list.append(data.get("obs", np.zeros((self.env_info["n_agents"], self.env_info["obs_shape"]))))
                state_list.append(data.get("state", np.zeros(self.env_info["state_shape"])))
                avail_actions_list.append(np.ones((self.env_info["n_agents"], self.env_info["n_actions"])))


                if not data["terminated"]:
                    all_terminated_this_step = False
                episode_lengths[i] = self.t + 1


            # Update batch with results of the step
            self.batch.update(post_transition_data, ts=self.t)
            
            self.t += 1
            env_steps_this_episode += self.batch_size # Assuming one step per env in batch

            # Update next step's pre-transition data (obs, state, avail_actions)
            # This is for t+1, so it's stored at index t+1 in the batch
            if not all_terminated_this_step : # Only if not all envs terminated
                 self.batch.update({
                    "state": state_list,
                    "avail_actions": avail_actions_list,
                    "obs": obs_list
                }, ts=self.t)


            if all_terminated_this_step:
                all_terminated = True # All parallel environments have finished their episode

        self.t_env += env_steps_this_episode # Update total env steps for this runner instance

        # Basic stats (can be expanded)
        # For MPE, "win_rate" might not be standard, but can be derived from info if applicable
        # Example: info might contain "is_success" or similar metrics.
        # For now, just return and length.
        avg_return = np.mean(episode_returns)
        avg_length = np.mean(episode_lengths)
        
        # This is a simplified stats_info. Real stats would come from env.get_stats() or info dicts.
        stats_info = {
            "episode_return": avg_return,
            "episode_length": avg_length,
            "win_rate": None, # MPE might not have a direct win_rate
            "stats": {} # Placeholder for other env-specific stats
        }
        
        # The runner in gumbel_online_train.py expects (episode_batch, mcts_buffer, stats_info)
        # mcts_buffer would contain trajectories for training the MCTS policy network.
        # This needs to be populated if MCTS is actually run.
        return self.batch, mcts_trajectory_data, stats_info


    def _batch_select_skill_with_mcts(self, env_indices, record_distribution=False):
        # Placeholder: Implement MCTS skill selection if using hierarchical policies
        # This is a STUB from the provided hier_mcts_parallel_runner.py
        pass

    def plot_distribution(self, save_dir="./results/mcts_distributions"):
        # Placeholder: Implement MCTS distribution plotting
        # This is a STUB from the provided hier_mcts_parallel_runner.py
        pass
        
    def _log(self, returns, stats, prefix):
        # Placeholder: Implement logging
        # This is a STUB from the provided hier_mcts_parallel_runner.py
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()


    # Other methods like save_initial_state, test_mcts_bandit, plot_bandit_stats
    # are kept as stubs from hier_mcts_parallel_runner.py
    def save_initial_state(self):
        pass

    def test_mcts_bandit(self, num_trials=1):
        pass

    def plot_bandit_stats(self, save_dir="./results/mcts_bandit"):
        pass
