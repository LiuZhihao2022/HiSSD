REGISTRY = {}

# normal agents
from .rnn_agent import RNNAgent
from .multi_task.hissd_agent import HISSDAgent
from .single_task.hissd_agent import HISSDAgent as HISSDAgentSingleTask 
REGISTRY["rnn"] = RNNAgent
REGISTRY["mt_hissd"] = HISSDAgent
REGISTRY["st_hissd"] = HISSDAgentSingleTask
