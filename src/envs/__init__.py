from functools import partial

from .multiagentenv import MultiAgentEnv
from .matrix_game.cts_matrix_game import Matrixgame as CtsMatrix
from .particle import Particle
# from .multiagent_particle_envs.make_env import make_env
from .multiagent_particle_envs.multiagent.mpe_env import MPE_env
# from .mamujoco import ManyAgentAntEnv, ManyAgentSwimmerEnv, MujocoMulti
# from smac.env import MultiAgentEnv, StarCraft2Env


def env_fn(env, **kwargs) -> MultiAgentEnv:
    # env_args = kwargs.get("env_args", {})
    return env(**kwargs)

# def env_fn2(env, **kwargs) -> MultiAgentEnv:
#
#     return env(**kwargs)



REGISTRY = {}
REGISTRY["cts_matrix_game"] = partial(env_fn, env=CtsMatrix)
REGISTRY["particle"] = partial(env_fn, env=Particle)
REGISTRY["mpe"] = partial(env_fn, env=MPE_env)


# REGISTRY["mujoco_multi"] = partial(env_fn, env=MujocoMulti)
# REGISTRY["manyagent_swimmer"] = partial(env_fn, env=ManyAgentSwimmerEnv)
# REGISTRY["manyagent_ant"] = partial(env_fn, env=ManyAgentAntEnv)
# REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
