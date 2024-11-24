# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import torch
import torch.nn.functional as F
import random

from isaacgym import gymtorch
from isaacgym import gymapi


from isaacgymenvs.utils.torch_jit_utils import to_torch, get_axis_params, torch_rand_float, quat_rotate, quat_rotate_inverse,tensor_clamp
from isaacgymenvs.tasks.base.vec_task import VecTask

from typing import Tuple, Dict
import math


class Biped(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        self.cfg = cfg
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]
        self.pos_reward_scale = self.cfg["env"]["posRewardScale"]
        self.orient_reward_scale = self.cfg["env"]["orientRewardScale"]
        self.lin_vel_reward_scale = self.cfg["env"]["linVelRewardScale"]
        self.ang_vel_reward_scale = self.cfg["env"]["angVelRewardScale"]
        self.force_reward_scale = self.cfg["env"]["forceRewardScale"]
        self.action_scale = self.cfg["env"]["actionScale"]

        self.contact_force_cost = self.cfg["env"]["contactForceCost"]
        self.joints_at_limit_cost = self.cfg["env"]["jointsAtLimitCost"]
        self.death_cost = self.cfg["env"]["deathCost"]

        # target_p = self.cfg["env"]["targetP"]
        # target_v = self.cfg["env"]["targetV"]
        # self.target_states = target_p + target_v

        self.termination_height = self.cfg["env"]["terminationHeight"]
        self.initial_height_min = self.cfg["env"]["startingHeightMin"]
        self.initial_height_max = self.cfg["env"]["startingHeightMax"]
        self.target_height = self.cfg["env"]["targetHeight"]

        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.cfg["env"]["numObservations"] = 58
        self.cfg["env"]["numActions"] = 8

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        torques_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        sensors_per_env = 4

        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)  # shape: num_envs, num_bodies, xyz axis
        self.dof_force_sensor_tensor = gymtorch.wrap_tensor(torques_tensor).view(self.num_envs, self.num_dof)

        self.target_states = torch.zeros_like(self.root_states,device=self.device, dtype=torch.float)
        self.target_states[:,2] = self.target_height

        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:, 0:13] = 0
        self.initial_root_states[:, 6] = 1
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.initial_dof_pos = torch.zeros_like(self.dof_pos, device=self.device, dtype=torch.float)
        self.initial_dof_vel = torch.zeros_like(self.dof_vel, device=self.device, dtype=torch.float)

        zero_tensor = torch.tensor([0.0], device=self.device)
        self.initial_dof_pos = torch.where(self.dof_limits_lower > zero_tensor, self.dof_limits_lower,
                                           torch.where(self.dof_limits_upper < zero_tensor, self.dof_limits_upper, self.initial_dof_pos))

        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.reset_idx(torch.arange(self.num_envs, device=self.device))


    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))
        # If randomizing, apply once immediately on startup before the fist sim step

        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file = "urdf/biped/urdf/biped.urdf"

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 10.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        biped_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(biped_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(biped_asset)
        self.num_joints = self.gym.get_asset_joint_count(biped_asset)

        left_coxa_idx = self.gym.find_asset_rigid_body_index(biped_asset, "left_coxa")
        left_femur_idx = self.gym.find_asset_rigid_body_index(biped_asset, "left_femur")
        left_tibia_idx = self.gym.find_asset_rigid_body_index(biped_asset, "tibia")
        left_wheel_idx = self.gym.find_asset_rigid_body_index(biped_asset, "wheel")

        right_coxa_idx = self.gym.find_asset_rigid_body_index(biped_asset, "right_coxa")
        right_femur_idx = self.gym.find_asset_rigid_body_index(biped_asset, "right_femur")
        right_tibia_idx = self.gym.find_asset_rigid_body_index(biped_asset, "tibia_2")
        right_wheel_idx = self.gym.find_asset_rigid_body_index(biped_asset, "wheel_2")

        left_wheel_dof_idx = self.gym.find_asset_dof_index(biped_asset,"wheel_l")
        right_wheel_dof_idx = self.gym.find_asset_dof_index(biped_asset,"wheel_r")

        # torso_idx = self.gym.find_asset_rigid_body_index(biped_asset, "torso")
        torso_idx = 0

        sensor_pose = gymapi.Transform()
        self.gym.create_asset_force_sensor(biped_asset, left_tibia_idx, sensor_pose)
        self.gym.create_asset_force_sensor(biped_asset, right_tibia_idx, sensor_pose)
        self.gym.create_asset_force_sensor(biped_asset, left_wheel_idx, sensor_pose)
        self.gym.create_asset_force_sensor(biped_asset, right_wheel_idx, sensor_pose)
        self.gym.create_asset_force_sensor(biped_asset, torso_idx, sensor_pose)

        self.actor_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        dof_props = self.gym.get_asset_dof_properties(biped_asset)
        for i in range(self.num_dof):
            dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            dof_props['stiffness'][i] = self.cfg["env"]["control"]["stiffness"] #self.Kp
            dof_props['damping'][i] = self.cfg["env"]["control"]["damping"] #self.Kd
            if dof_props['lower'][i] > dof_props['upper'][i]:
                self.dof_limits_lower.append(dof_props['upper'][i])
                self.dof_limits_upper.append(dof_props['lower'][i])
            else:
                self.dof_limits_lower.append(dof_props['lower'][i])
                self.dof_limits_upper.append(dof_props['upper'][i])
                
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            height = random.uniform(self.initial_height_min, self.initial_height_max)

            start_pose = gymapi.Transform()
            start_pose.p = gymapi.Vec3(0.0, 0.0, height)
            start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

            handle = self.gym.create_actor(env_ptr, biped_asset, start_pose, "biped", i, 1)
            self.gym.set_actor_dof_properties(env_ptr, handle, dof_props)
            self.gym.enable_actor_dof_force_sensors(env_ptr, handle)
            # self.gym.set_actor_scale(env_ptr, handle, 1)

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(
                    env_ptr, handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0, 0.5, 1.0))
                
            self.envs.append(env_ptr)
            self.actor_handles.append(handle)
        

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)
    
    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        targets = self.action_scale * self.actions
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(targets))

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

    def compute_reward(self, actions):
        
        up_vector_points = torch.zeros(self.num_envs,3,device=self.device)
        up_vector_points[:,2] = 1
        rotation_matrices = self.quaternion_to_rotation_matrix(self.root_states[:,3:7])
        rotated_points = torch.matmul(rotation_matrices, up_vector_points.unsqueeze(-1)).squeeze(-1)

        rotated_points_normalized = torch.nn.functional.normalize(rotated_points,p=2,dim=1)
        dot_products = torch.sum(up_vector_points * rotated_points_normalized, dim=1)
        self.angle_radians = torch.acos(dot_products)

        self.rew_buf[:], self.reset_buf[:] = compute_biped_reward(self.root_states,
                                                     self.target_states,
                                                     
                                                     self.angle_radians,
                                                     self.dof_force_sensor_tensor,
                                                     self.contact_forces,
                                                     self.dof_limits_lower,
                                                     self.dof_limits_upper,

                                                     self.termination_height,
                                                     self.progress_buf,
                                                     self.max_episode_length,
                                                     self.death_cost,
                                                     self.joints_at_limit_cost,

                                                     self.pos_reward_scale,
                                                     self.orient_reward_scale,
                                                     self.lin_vel_reward_scale,
                                                     self.ang_vel_reward_scale,
                                                     self.force_reward_scale,
                                                     self.num_envs
        )
        # print(self.reset_buf)

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)  # done in step
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        self.obs_buf[:] = compute_biped_observations(self.root_states,
                                                     self.target_states,
                                                     self.dof_pos,
                                                     self.dof_vel,
                                                     self.dof_force_sensor_tensor,
                                                     self.actions
        )

    def reset_idx(self, env_ids):
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        height = random.uniform(self.initial_height_min, self.initial_height_max)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(0.0, 0.0, height)
        self.initial_root_states[:,2] = height

        positions = torch_rand_float(-0.5, 0.5, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.5, 0.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_pos[env_ids] = tensor_clamp(self.initial_dof_pos[env_ids] + positions, self.dof_limits_lower, self.dof_limits_upper)
        self.dof_vel[env_ids] = velocities
        
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.initial_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # print(self.progress_buf)

    def quaternion_to_rotation_matrix(self, quaternion):
        # Normalize quaternion
        quaternion = F.normalize(quaternion, p=2, dim=-1)
        
        # Extract components
        x, y, z, w = quaternion[:, 0], quaternion[:, 1], quaternion[:, 2], quaternion[:, 3]
        
        # Compute rotation matrix
        rotation_matrix = torch.stack([
            1 - 2*y**2 - 2*z**2,  2*x*y - 2*w*z,      2*x*z + 2*w*y,
            2*x*y + 2*w*z,       1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x,
            2*x*z - 2*w*y,       2*y*z + 2*w*x,      1 - 2*x**2 - 2*y**2
        ], dim=-1).view(-1, 3, 3)
        
        return rotation_matrix

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_biped_reward(root_states,
                         target_state,

                         angle_radians,
                         dof_forces,
                         contact_forces,
                         dof_limits_lower,
                         dof_limits_upper,

                         termination_height,
                         progress_buff,
                         max_ep_length,
                         death_cost,
                         joint_at_limit_cost,

                         pos_reward_scale,
                         orient_reward_scale,
                         lin_vel_reward_scale,
                         ang_vel_reward_scale,
                         force_reward_scale,
                         num_envs
                         ):
    

    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, int, float, float, float, float, float, float, float, int)   -> Tuple[Tensor, Tensor]
    base_index = 0

    pos_error = torch.sum(torch.square(root_states[:,0:3] - target_state[:,0:3]),dim=1)
    orient_error = torch.square(angle_radians)
    lin_vel_error = torch.sum(torch.square(root_states[:,7:10] - target_state[:,7:10]),dim=1)
    ang_vel_error = torch.sum(torch.square(root_states[:,10:13] - target_state[:,10:13]),dim=1)
    
    rew_pos = torch.exp(-pos_error/4.0) * pos_reward_scale
    rew_orient = torch.exp(-orient_error/10) * orient_reward_scale
    rew_lin_vel = torch.exp(-lin_vel_error/0.25) * lin_vel_reward_scale
    rew_ang_vel = torch.exp(-ang_vel_error/0.25) * ang_vel_reward_scale

    rew_torque = torch.sum(torch.square(dof_forces),dim=1) * force_reward_scale
    
    # total_reward = rew_pos + rew_lin_vel + rew_ang_vel + rew_torque
    total_reward = rew_pos + rew_orient + rew_lin_vel + rew_ang_vel + rew_torque
    
    reset = torch.where(root_states[:,2] < termination_height, 1, 0)
    reset = reset | torch.any(torch.norm(contact_forces[:,base_index,:], dim=1) > 0.0, dim=0)

    time_out = torch.zeros(num_envs, dtype=torch.int, device=root_states.device)  # Initialize timeout tensor with zeros

    # time_out[progress_buff >= (max_ep_length-1)]  # no terminal reward for time-outs
    time_out = torch.where(progress_buff > max_ep_length-1, 1, 0)
    reset = reset | time_out
    # print(f"time_out {time_out}, progress: {progress_buff}")
    # print(f"z height: \n{root_states[:,2]} \n reset: \n{reset}")
    # print(f"orient - target:\n {torch.square(root_states[:,3:7] - target_state[:,3:7])}")
    # print(total_reward)
    # print(f"pos:\n{rew_pos}\nangle:\n{rew_orient}\nlin_vel:\n{rew_lin_vel}\nang_vel:\n{rew_ang_vel}\ntorque:\n{rew_torque}")
    return total_reward.detach(), reset


@torch.jit.script
def compute_biped_observations(root_states,
                               target_states,

                               dof_positions,
                               dof_velocities,
                               dof_forces,

                               actions
                               ):

    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor

    to_target = target_states - root_states

    obs = torch.cat((root_states,
                     to_target,
                     dof_positions,
                     dof_velocities,
                     dof_forces,
                     actions
                    ), dim=-1)

    return obs



