from .episode_runner import EpisodeRunner as MultiTaskEpisodeRunner
from .episode_ada_runner import AdaEpisodeRunner as MultiTaskAdaEpisodeRunner
from .parallel_runner import ParallelRunner as MultiTaskParallelRunner
from .hier_mcts_parallel_runner import HierMCTSParallelRunner as MultiTaskHierMCTSParallelRunner
from .hier_mcts_episode_runner import HierMCTSEpisodeRunner as MultiTaskHierMCTSEpisodeRunner
from .episode_runner import EpisodeRunner
from .hier_mcts_episode_runner import HierMCTSEpisodeRunner

REGISTRY = {}

REGISTRY["mt_episode"] = MultiTaskEpisodeRunner
REGISTRY["mt_ada_episode"] = MultiTaskAdaEpisodeRunner
REGISTRY["mt_parallel"] = MultiTaskParallelRunner
REGISTRY["mt_hier_mcts_parallel"] = MultiTaskHierMCTSParallelRunner
REGISTRY["mt_hier_mcts_episode"] = MultiTaskHierMCTSEpisodeRunner
REGISTRY["episode"] = EpisodeRunner
REGISTRY["hier_mcts_episode"] = HierMCTSEpisodeRunner