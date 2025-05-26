import sys
sys.path.append("/home/lzh/HiSSD")
from functools import partial
from smac.env import MultiAgentEnv, StarCraft2Env
from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper
from .gymma import GymmaWrapper
from .mpe_env_wrapper import MPEEnvWrapper
from OMAR.multiagent_particle_envs.omar_env import OMAREnv


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


def gymma_fn(env, **kwargs) -> MultiAgentEnv:
    assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
    return env(**kwargs)


REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["sc2v2"] = partial(env_fn, env=StarCraftCapabilityEnvWrapper)
REGISTRY["gymma"] = partial(gymma_fn, env=GymmaWrapper)
REGISTRY["mpe"] = partial(env_fn, env=OMAREnv)
REGISTRY["test"] = StarCraftCapabilityEnvWrapper
