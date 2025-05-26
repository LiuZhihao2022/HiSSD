REGISTRY = {}

from .basic_controller import BasicMAC
from .multi_task.mt_hissd_controller import HISSDSMAC
from .single_task.hissd_controller import HISSDSMAC as HISSDSMACSingleTask
REGISTRY["basic_mac"] = BasicMAC
REGISTRY["mt_hissd_mac"] = HISSDSMAC
REGISTRY["st_hissd_mac"] = HISSDSMACSingleTask