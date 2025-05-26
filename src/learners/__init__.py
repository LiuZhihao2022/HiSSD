REGISTRY = {}

# normal learner
from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .multi_task.hissd_learner import HISSDLearner
from .single_task.hissd_learner import HISSDLearner as HISSDLearnerSingleTask
from .multi_task.odis_learner import ODISLearner
from .multi_task.ma_gumbel_learner import MAGumbelLearner
from .multi_task.hissd_learner_continue_train import HISSDLearnerContinueTrain
REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["st_hissd_learner"] = HISSDLearnerSingleTask
REGISTRY["hissd_learner"] = HISSDLearner
REGISTRY["odis_learner"] = ODISLearner
REGISTRY["ma_gumbel_learner"] = MAGumbelLearner
REGISTRY["hissd_learner_continue_train"] = HISSDLearnerContinueTrain
