# Ref: source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/cartpole/__init__.py

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Simplified catch thrown environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-ArmHand-Simplified-CatchThrown-Direct-v0",
    entry_point=f"{__name__}.armhand_simplified_catchthrown_env:ArmHandSimplifiedCatchThrownEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.armhand_simplified_catchthrown_env:ArmHandSimplifiedCatchThrownEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)