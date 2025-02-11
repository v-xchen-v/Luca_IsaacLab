# Ref: https://isaac-sim.github.io/IsaacLab/main/source/how-to/write_articulation_cfg.html
# Ref: source/extensions/omni.isaac.lab_assets/omni/isaac/lab_assets/franka.py

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka Emika robots.

The following configurations are available:

* :obj:`FRANKA_PANDA_CFG`: Franka Emika Panda robot with Panda hand
* :obj:`FRANKA_PANDA_HIGH_PD_CFG`: Franka Emika Panda robot with Panda hand with stiffer PD control
"""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##

REALMAN_XHAND_R_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/yichao/Documents/repos/Luca_Data/Robots/Realman_XHand/realman_xhand_right.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1": 0.0,
            "joint2": -0.569,
            "joint3": 0.0,
            "joint4": -2.210,
            "joint5": 0.0,
            "joint6": 2.037,
            "joint7": 0.741,
            "right_hand_index_bend_joint": 0,
            "right_hand_mid_joint1": 0, 
            "right_hand_pinky_joint1": 0, 
            "right_hand_ring_joint1": 0, 
            "right_hand_thumb_bend_joint": 0,
            "right_hand_index_joint1": 0, 
            "right_hand_index_joint2": 0, 
            "right_hand_mid_joint2": 0, 
            "right_hand_pinky_joint2": 0, 
            "right_hand_ring_joint2": 0, 
            "right_hand_thumb_rota_joint1": 0, 
            "right_hand_thumb_rota_joint2": 0,
        },
    ),
    actuators={
        # "realman_shoulder": ImplicitActuatorCfg(
        #     joint_names_expr=["joint[1-4]"],
        #     effort_limit=87.0,
        #     velocity_limit=2.175,
        #     stiffness=80.0,
        #     damping=4.0,
        # ),
        # "realman_forearm": ImplicitActuatorCfg(
        #     joint_names_expr=["joint[5-7]"],
        #     effort_limit=12.0,
        #     velocity_limit=2.61,
        #     stiffness=80.0,
        #     damping=4.0,
        # ),
        "all": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-7]", "right_hand.*"],
            effort_limit=12.0,
            velocity_limit=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Franka Emika Panda robot."""


# REALMAN_HIGH_PD_CFG = REALMAN_CFG.copy()
# REALMAN_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
# REALMAN_HIGH_PD_CFG.actuators["realman_shoulder"].stiffness = 400.0
# REALMAN_HIGH_PD_CFG.actuators["realman_shoulder"].damping = 80.0
# REALMAN_HIGH_PD_CFG.actuators["realman_forearm"].stiffness = 400.0
# REALMAN_HIGH_PD_CFG.actuators["realman_forearm"].damping = 80.0
"""Configuration of Franka Emika Panda robot with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""
