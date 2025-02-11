# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

from omni.isaac.lab_assets.cartpole import CARTPOLE_CFG
from omni.isaac.lab_assets.realman import REALMAN_CFG
from omni.isaac.lab_assets.realman_xhand_R import REALMAN_XHAND_R_CFG

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import sample_uniform

# Then you’ll probably have to start the SimulationApp() before importing those packages
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.lab.assets import RigidObjectCfg, RigidObject

@configclass
class ArmHandSimplifiedCatchThrownEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    action_scale = 100.0  # [N]
    action_space = 7+12 # 7 joints for the robot arm and 12 joints for the hand
    observation_space = 7+7+3
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = REALMAN_XHAND_R_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # cart_dof_name = "realman_all"
    # pole_dof_name = "realman_tmp"

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # reset
    max_cart_pos = 2.0  # the cart is reset if it exceeds that position [m]
    initial_pole_angle_range = [-0.25, 0.25]  # the range in which the pole angle is sampled from on reset [rad]

    # reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_pole_pos = -1.0
    rew_scale_cart_vel = -0.01
    rew_scale_pole_vel = -0.005


class ArmHandSimplifiedCatchThrownEnv(DirectRLEnv):
    cfg: ArmHandSimplifiedCatchThrownEnvCfg

    def __init__(self, cfg: ArmHandSimplifiedCatchThrownEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # self._cart_dof_idx, _ = self.cartpole.find_joints(self.cfg.cart_dof_name)
        # self._pole_dof_idx, _ = self.cartpole.find_joints(self.cfg.pole_dof_name)
        
        # Ref: source/extensions/omni.isaac.lab/omni/isaac/lab/assets/articulation/articulation.py
        dof_names = [
            "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7",
            "right_hand_index_bend_joint",
            "right_hand_mid_joint1", 
            "right_hand_pinky_joint1", 
            "right_hand_ring_joint1", 
            "right_hand_thumb_bend_joint",
            "right_hand_index_joint1", 
            "right_hand_index_joint2", 
            "right_hand_mid_joint2", 
            "right_hand_pinky_joint2", 
            "right_hand_ring_joint2", 
            "right_hand_thumb_rota_joint1", 
            "right_hand_thumb_rota_joint2"
        ]
        self.dof_idx, self.dof_names = self.cartpole.find_joints(dof_names)
        self.action_scale = self.cfg.action_scale

        self.joint_pos = self.cartpole.data.joint_pos
        self.joint_vel = self.cartpole.data.joint_vel
        
        # splits to two steps
        self.step_stage = torch.ones((self.num_envs, 1)).to(self.device)# Start with step 1
        self.threshold_1 = 0.015 # Distance threshold for step 1
        
        self.num_arm_dof = 7

    def _setup_scene(self):
        self.cartpole = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        
        # add a ball
        # Ref: source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/shadow_hand/shadow_hand_env_cfg.py
  
        # Rigid Object
        cone_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Sphere",
            spawn=sim_utils.SphereCfg(
                radius=0.04,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.16),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(-1.5, 0.0, 0.02), rot=(1.0, 0.0, 0.0, 0.0)),
        )
        self.sphere_object = RigidObject(cfg=cone_cfg)
        
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["cartpole"] = self.cartpole
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        



    def _pre_physics_step(self, actions: torch.Tensor) -> None:

        
        self.actions = self.action_scale * actions.clone()


    def _apply_action(self) -> None:
        # if self.step_stage == 1:
        #     self.actions[-12:] = 0.0
        # else:
        #     self.actions[:7] = 0.0
        
        # self.step_stage is [num_envs, 1], got where the value is 1, can set the self.actions[idx,self.num_arm_dof:] to 0
        self.actions[self.step_stage[:, 0] == 1, self.num_arm_dof:] = 0.0
        
        self.cartpole.set_joint_effort_target(self.actions, joint_ids=self.dof_idx)

    def _get_observations(self) -> dict:
        # obs = torch.cat(
        #     (
        #         self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
        #         self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
        #         self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
        #         self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
        #     ),
        #     dim=-1,
        # )
        # random observation
        num_envs = self.joint_pos.shape[0]
        obs_shape = (num_envs, 4)
        obs = torch.rand(obs_shape)
        
        # real observation
        # - sphere position and linear velocity
        # - robot arm end effector position and velocity
        # - robot arm end effector and sphere relative position
        
        # get sphere position and velocity
        self.sphere_pos = self.sphere_object.data.root_link_state_w[:, :3].clone() # position
        # get pos relative to environment origins
        self.sphere_pos -= self.scene.env_origins
        
        self.sphere_vel = self.sphere_object.data.root_link_state_w[:, 7:11] # linear velocity
        
        # get robot arm end effector position and velocity
        # end_effector_pos = self.joint_pos[:, self.dof_idx[-1]] # position
        # end_effector_vel = self.joint_vel[:, self.dof_idx[-1]] # velocity
        # link7 is the end effector
        self.ee_idx, self.ee_name = self.cartpole.find_bodies("Link7")
        # self.cartpole.data.body_state_w [num_envs, num_bodies, 13], 13: [pos, rot, lin_vel, ang_vel]
        self.end_effector_pos = self.cartpole.data.body_state_w[:, self.ee_idx, :3].clone() # position
        # reshape self.end_effector_pos from [num_envs, 1, 3] to [num_envs, 3]
        self.end_effector_pos = self.end_effector_pos.squeeze(dim=1)
        # get pos relative to environment origins
        self.end_effector_pos -= self.scene.env_origins
        self.end_effector_vel = self.cartpole.data.body_state_w[:, self.ee_idx, 7:11] # linear velocity
        self.end_effector_vel = self.end_effector_vel.squeeze(dim=1)
        # end_effector_pos = end_effector[0].get_pose()[:3]
        
        # get robot arm end effector and sphere relative position

        relative_pos = self.sphere_pos - self.end_effector_pos
        
        # stack all the observations
        obs = torch.cat((self.sphere_pos, self.sphere_vel, self.end_effector_pos, self.end_effector_vel, relative_pos), dim=-1)
        
        # distance of ee and ball to determine step stage
        distance = torch.norm(self.sphere_pos - self.end_effector_pos, dim=1)
        self.step_stage[distance < self.threshold_1, 0] = torch.tensor(2, dtype=torch.float32).to(self.device) # if distance < threshold, then step 2, else step 1
        
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_pole_pos,
            self.cfg.rew_scale_cart_vel,
            self.cfg.rew_scale_pole_vel,
            self.sphere_pos,
            self.sphere_vel,
            self.end_effector_pos,
            self.end_effector_vel,
            self.reset_terminated,
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.cartpole.data.joint_pos
        self.joint_vel = self.cartpole.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self.dof_idx]) > self.cfg.max_cart_pos, dim=1)
        # if ball out of bounds, then terminate the episode
        self.sphere_object.update(self.physics_dt)
        sphere_pos = self.sphere_object.data.root_link_state_w[:, :3].clone()
        sphere_pos = sphere_pos-self.scene.env_origins
        out_of_bounds = torch.any(torch.abs(sphere_pos) > self.cfg.max_cart_pos, dim=1)
        # out_of_bounds = torch.zeros_like(out_of_bounds)
        # out_of_bounds = out_of_bounds | torch.any(torch.abs(self.joint_pos[:, self._pole_dof_idx]) > math.pi / 2, dim=1)
        
        # if the sphere is intercepted by the end effector, then terminate the episode
        distance = torch.norm(sphere_pos - self.end_effector_pos, dim=1)
        intercepted = distance < 0.1
        out_of_bounds = intercepted | out_of_bounds
        
        # if the sphere's x is bigger than the end effector's x, then terminate the episode
        miss_ee = sphere_pos[:, 0] > 0 #self.end_effector_pos[:, 0]
        out_of_bounds = miss_ee | out_of_bounds
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.cartpole._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.cartpole.data.default_joint_pos[env_ids]
        joint_pos[:, self.dof_idx] += sample_uniform(
            self.cfg.initial_pole_angle_range[0] * math.pi,
            self.cfg.initial_pole_angle_range[1] * math.pi,
            joint_pos[:, self.dof_idx].shape,
            joint_pos.device,
        )
        joint_vel = self.cartpole.data.default_joint_vel[env_ids]

        default_root_state = self.cartpole.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.cartpole.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self.cartpole.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.cartpole.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
        
        # reset the sphere as default position
        # reset object
        object_default_state = self.sphere_object.data.default_root_state.clone()[env_ids]
        # pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)

        object_default_state[:, 0:3] = (
            object_default_state[:, 0:3] + self.scene.env_origins[env_ids]
        )

        # rot_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)  # noise for X and Y rotation
        # object_default_state[:, 3:7] = randomize_rotation(
        #     rot_noise[:, 0], rot_noise[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        # )

        object_default_state[:, 7:] = torch.zeros_like(self.sphere_object.data.default_root_state[env_ids, 7:])
        self.sphere_object.write_root_link_pose_to_sim(object_default_state[:, :7], env_ids)
        self.sphere_object.write_root_com_velocity_to_sim(object_default_state[:, 7:], env_ids)
        
        # Ref: source/standalone/playground/throw_a_sphere.py
        # throw the ball by apply a force at the beginning
        body_ids, body_names = self.sphere_object.find_bodies(".*")
        external_wrench_b = torch.zeros(self.sphere_object.num_instances, len(body_ids), 6, device=joint_pos.device)
        # Every 2nd cube should have a force applied to it
        external_wrench_b[env_ids, :, 0] = 9.81 * self.sphere_object.root_physx_view.get_masses()[0]*50
        external_wrench_b[env_ids, :, 2] = 9.81 * self.sphere_object.root_physx_view.get_masses()[0]*45
        self.sphere_object.set_external_force_and_torque(
            forces=external_wrench_b[..., :3], 
            torques=external_wrench_b[..., 3:], body_ids=body_ids)

        # apply sim data
        self.sphere_object.write_data_to_sim()


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_pole_pos: float,
    rew_scale_cart_vel: float,
    rew_scale_pole_vel: float,
    sphere_pos: torch.Tensor,
    sphere_vel: torch.Tensor,
    ee_pos: torch.Tensor,
    ee_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
    # step_stage: int,
    # step_threshold: float,
):
    # rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    # rew_termination = rew_scale_terminated * reset_terminated.float()
    # rew_pole_pos = rew_scale_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(dim=1), dim=-1)
    # rew_cart_vel = rew_scale_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(dim=1), dim=-1)
    # rew_pole_vel = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
    # total_reward = rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel
    
    # # a ramdom reward
    # num_envs = pole_pos.shape[0]
    # total_reward = torch.rand(num_envs)
    
    # real reward
    # - position reward of reducing the distance between the sphere and the end effector
    # - a larger reward if the sphere is successfully intercepted by the end effector
    # - a smaller reward if the sphere is not intercepted by the end effector    
    total_reward = torch.zeros_like(reset_terminated)
    # get distance between sphere and end effector
    distance = torch.norm(sphere_pos - ee_pos, dim=1)
    
    # if step_stage == 1 and distance < step_threshold:
    #     step_stage = 2
        
    # determine if the sphere is intercepted by the end effector
    intercepted = distance < 0.1
    intercepted_reward = torch.where(intercepted, torch.tensor(1.0), torch.tensor(-1.0))
    
    total_reward = intercepted_reward  + distance*-1.0
    return total_reward
