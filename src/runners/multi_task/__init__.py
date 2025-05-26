from .episode_runner import EpisodeRunner as MultiTaskEpisodeRunner
from .episode_ada_runner import AdaEpisodeRunner as MultiTaskAdaEpisodeRunner
from .parallel_runner import ParallelRunner as MultiTaskParallelRunner
from .hier_mcts_parallel_runner import HierMCTSParallelRunner
from .mpe_hier_mcts_parallel_runner import MPEHierMCTSParallelRunner
from .episode_runner import EpisodeRunner
from .hier_mcts_episode_runner import HierMCTSEpisodeRunner
from .test_mcts_parallel_runner import TestMCTSParallelRunner
from .hier_mcts_parallel_runner_continue_train import HierMCTSParallelRunnerContinueTrain
from ..single_task.hier_mcts_parallel_runner_single_task import HierMCTSParallelRunner as HierMCTSParallelRunnerSingleTask
REGISTRY = {}

REGISTRY["mt_episode"] = MultiTaskEpisodeRunner
REGISTRY["mt_ada_episode"] = MultiTaskAdaEpisodeRunner
REGISTRY["mt_parallel"] = MultiTaskParallelRunner
REGISTRY["hier_mcts_parallel"] = HierMCTSParallelRunner
REGISTRY["episode"] = EpisodeRunner
REGISTRY["hier_mcts_episode"] = HierMCTSEpisodeRunner
REGISTRY["test_mcts_parallel"] = TestMCTSParallelRunner
REGISTRY["hier_mcts_parallel_continue_train"] = HierMCTSParallelRunnerContinueTrain
REGISTRY["mpe_hier_mcts_parallel"] = MPEHierMCTSParallelRunner
REGISTRY["hier_mcts_parallel_single_task"] = HierMCTSParallelRunnerSingleTask