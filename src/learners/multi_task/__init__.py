from .odis_learner import ODISLearner
from .hissd_learner import HISSDLearner
from .ma_gumbel_learner import MAGumbelLearner
from .hissd_learner_continue_train import HISSDLearnerContinueTrain
REGISTRY = {}

REGISTRY["odis_learner"] = ODISLearner
REGISTRY["hissd_learner"] = HISSDLearner
REGISTRY["ma_gumbel_learner"] = MAGumbelLearner
REGISTRY["hissd_learner_continue_train"] = HISSDLearnerContinueTrain