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
import time

class HexapodTest(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        self.cfg = cfg
        self.randomize= self.cfg["task"]["randomize"]
        
        # reward scales
        self.rew_scales={}
        self.rew_scales["lin_vel_xy_rew_scale"] = self.cfg["env"]["learn"]["linearVelocityXYRewardScale"]
        self.rew_scales["ang_vel_z_rew_scale"] = self.cfg["env"]["learn"]["angularVelocityZRewardScale"]
        self.rew_scales["joint_central_reward"] = self.cfg["env"]["learn"]["jointCentralReward"]
        

        # normalization
        self.lin_vel_scale = self.cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self.cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self.cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self.cfg["env"]["learn"]["dofVelocityScale"]
        self.torque_scale = self.cfg["env"]["learn"]["torqueScale"]

        # costs
        self.costs = {}
        self.costs["lin_vel_z_cost"] = self.cfg["env"]["learn"]["linearVelocityzRewardCost"]
        self.costs["ang_vel_xy_cost"] = self.cfg["env"]["learn"]["angularVelocityXYRewardCost"]
        self.costs["torque_cost"] = self.cfg["env"]["learn"]["torqueRewardCost"]
        self.costs["leg_collision_cost"] = self.cfg["env"]["learn"]["legCollisionRewardCost"]
        self.costs["orientation_cost"] = self.cfg["env"]["learn"]["orientationRewardCost"]
        self.costs["height_cost"] = self.cfg["env"]["learn"]["heightRewardCost"]
        self.costs["joint_acc"] = self.cfg["env"]["learn"]["jointAccelerationRewardCost"]
        self.costs["feet_air_time_cost"] = self.cfg["env"]["learn"]["feetAirTimeRewardCost"]


        # command ranges 
        self.command_x_range = self.cfg["env"]["randomCommandRanges"]["linear_x"]
        self.command_y_range = self.cfg["env"]["randomCommandRanges"]["linear_y"]
        self.command_yaw_range = self.cfg["env"]["randomCommandRanges"]["yaw"]
        self.command_height_range = self.cfg["env"]["randomCommandRanges"]["commandHeightRange"]
        self.action_scale = self.cfg["env"]["control"]["actionScale"]
        
        # base init state
        pos = self.cfg["env"]["baseInitState"]["pos"]
        rot = self.cfg["env"]["baseInitState"]["rot"]
        v_lin = self.cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self.cfg["env"]["baseInitState"]["vAngular"]
        self.base_init_state = pos + rot + v_lin + v_ang

        # default joint positions
        self.named_default_joint_angles = self.cfg["env"]["defaultJointAngles"]

        # plane parameters
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        #other
        self.max_episode_length = self.cfg["env"]["episodeLength"]

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        torques_sensor_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        net_contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.actor_root_states = gymtorch.wrap_tensor(actor_root_state_tensor)
        self.dof_states = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_states.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_states.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.initial_dof_pos = torch.zeros_like(self.dof_pos, device=self.device, dtype=torch.float)
        self.torques_sensors = gymtorch.wrap_tensor(torques_sensor_tensor).view(self.num_envs, self.num_dof)
        self.net_contact_forces = gymtorch.wrap_tensor(net_contact_force_tensor).view(self.num_envs, -1, 3)

        self.last_dof_vel = self.dof_vel.clone()

        self.Kp = self.cfg["env"]["control"]["stiffness"]
        self.Kd = self.cfg["env"]["control"]["damping"]

        # for key in self.rew_scales.keys():
        #     self.rew_scales[key] *= self.dt

        if self.viewer != None:
            p = self.cfg["env"]["viewer"]["pos"]
            lookat = self.cfg["env"]["viewer"]["lookat"]
            cam_pos = gymapi.Vec3(p[0], p[1], p[2])
            cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        self.commands = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_y = self.commands.view(self.num_envs, 4)[..., 1]
        self.commands_x = self.commands.view(self.num_envs, 4)[..., 0]
        self.commands_yaw = self.commands.view(self.num_envs, 4)[..., 2]
        self.commands_height = self.commands.view(self.num_envs, 4)[..., 3]
        self.default_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False)

        for i in range(self.cfg["env"]["numActions"]):
            name = self.dof_names[i]
            angle = self.named_default_joint_angles[name]
            self.default_dof_pos[:,i] = angle
        
        self.extras = {}
        self.initial_root_states = self.actor_root_states.clone()
        self.initial_root_states[:] = to_torch(self.base_init_state, device=self.device, requires_grad=False)
        self.gravity_vec = to_torch(get_axis_params(-1.0,self.up_axis_index), device=self.device).repeat((self.num_envs,1))
        self.reset_idx(torch.arange(self.num_environments, device=self.device))


    def create_sim(self):
        self.up_axis_index = 2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file = "urdf/hexapod_test/urdf/hexapod.urdf"
        
        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        # asset_options.convex_decomposition_from_submeshes = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.vhacd_enabled = True

        hexapod_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(hexapod_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(hexapod_asset)
        self.num_joints = self.gym.get_asset_joint_count(hexapod_asset)

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self.body_names = self.gym.get_asset_rigid_body_names(hexapod_asset)
        self.dof_names = self.gym.get_asset_dof_names(hexapod_asset)

        self.base_index = 0

        extremity_name = "e"
        feet_names = [s for s in self.body_names if extremity_name in s]
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_asset_rigid_body_index(hexapod_asset, feet_names[i])
        extremity_name = "c"
        coxa_names = [s for s in self.body_names if extremity_name in s]
        self.coxa_indices = torch.zeros(len(coxa_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(coxa_names)):
            self.coxa_indices[i] = self.gym.find_asset_rigid_body_index(hexapod_asset, coxa_names[i])
        extremity_name = "f"
        femur_names = [s for s in self.body_names if extremity_name in s]
        self.femur_indices = torch.zeros(len(femur_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(femur_names)):
            self.femur_indices[i] = self.gym.find_asset_rigid_body_index(hexapod_asset, femur_names[i])
        extremity_name = "t"
        tibia_names = [s for s in self.body_names if extremity_name in s]
        self.tibia_indices = torch.zeros(len(tibia_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(tibia_names)):
            self.tibia_indices[i] = self.gym.find_asset_rigid_body_index(hexapod_asset, tibia_names[i])

        self.feet_contact_times = torch.zeros(self.num_envs, len(self.feet_indices), 1, device=self.device, dtype = torch.float, requires_grad=False)
        self.feet_not_contact_times = torch.zeros(self.num_envs, len(self.feet_indices), 1, device=self.device, dtype = torch.float, requires_grad=False)
        self.actor_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        dof_props = self.gym.get_asset_dof_properties(hexapod_asset)
        for i in range(self.num_dof):
            dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            dof_props['stiffness'][i] = self.cfg["env"]["control"]["stiffness"]
            dof_props['damping'][i] = self.cfg["env"]["control"]["damping"]
            if dof_props['lower'][i] > dof_props['upper'][i]:
                self.dof_limits_lower.append(dof_props['upper'][i])
                self.dof_limits_upper.append(dof_props['lower'][i])
            else:
                self.dof_limits_lower.append(dof_props['lower'][i])
                self.dof_limits_upper.append(dof_props['upper'][i])
        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)
        
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            actor_handle = self.gym.create_actor(env_ptr, hexapod_asset, start_pose, "hexapod", i, 0)
            self.gym.set_actor_dof_properties(env_ptr, actor_handle, dof_props)
            self.gym.enable_actor_dof_force_sensors(env_ptr, actor_handle)

            for i in range(self.num_bodies):
                if i in self.feet_indices:
                    color = gymapi.Vec3(1, 0.3, 0.3)
                elif i in self.coxa_indices:
                    color = gymapi.Vec3(0.3, 1, 0)
                elif i in self.femur_indices:
                    color = gymapi.Vec3(0, 1, 0.58)
                elif i in self.tibia_indices:
                    color = gymapi.Vec3(0, 1, 0.9)
                else:
                    color = gymapi.Vec3(1, 0.984, 0)

                self.gym.set_rigid_body_color(
                    env_ptr, actor_handle, i, gymapi.MESH_VISUAL, color)

            self.envs.append(env_ptr)
            self.actor_handles.append(actor_handle)

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        self.actions = (self.actions+1)*(self.dof_limits_upper - self.dof_limits_lower)/2 + self.dof_limits_lower
        targets = self.actions + self.default_dof_pos
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(targets))

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observation()
        self.compute_reward(self.actions)

        self.last_dof_vel[:] = self.dof_vel
    
    def reset_idx(self, env_ids):
        
        positions = torch_rand_float(-0.00, 0.00, (len(env_ids), self.num_dof), device=self.device)
        velocities= torch_rand_float(-0.00, 0.00, (len(env_ids), self.num_dof), device=self.device)
        self.dof_pos[env_ids] = tensor_clamp(self.initial_dof_pos[env_ids]+positions, self.dof_limits_lower, self.dof_limits_upper)
        self.dof_vel[env_ids] = velocities

        self.initial_root_states[env_ids,2] = torch_rand_float(0.15, 0.2, (len(env_ids),1), device=self.device).squeeze()
        self.initial_root_states[env_ids,3:6] = torch_rand_float(-0.2, 0.2, (len(env_ids),3), device=self.device)
        norms = torch.norm(self.initial_root_states[env_ids,3:7], dim=1, keepdim=True)
        self.initial_root_states[env_ids, 3:7] /= norms
        self.initial_root_states[env_ids, 7:10] = torch_rand_float(-0.1, 0.1, (len(env_ids), 3), device=self.device) 
        self.initial_root_states[env_ids, 10:13] = torch_rand_float(-0.1, 0.1, (len(env_ids), 3), device=self.device) 

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.initial_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.dof_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
        self.commands_x[env_ids] = torch_rand_float(self.command_x_range[0], self.command_x_range[1], (len(env_ids),1), device=self.device).squeeze()
        self.commands_y[env_ids] = torch_rand_float(self.command_y_range[0], self.command_y_range[1], (len(env_ids),1), device=self.device).squeeze()
        self.commands_yaw[env_ids] = torch_rand_float(self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids),1), device=self.device).squeeze()
        self.commands_height[env_ids] = torch_rand_float(self.command_height_range[0], self.command_height_range[1], (len(env_ids),1), device=self.device).squeeze()
        self.feet_not_contact_times[env_ids_int32] = 0.0

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

    def compute_reward(self, action):
        
        base_quat = self.actor_root_states[:,3:7]
        base_lin_vel = quat_rotate_inverse(base_quat, self.actor_root_states[:,7:10])
        base_ang_vel = quat_rotate_inverse(base_quat, self.actor_root_states[:,10:13])
        projected_gravity = quat_rotate(base_quat, self.gravity_vec)

        lin_vel_error = torch.sum(torch.square(base_lin_vel[:,:2] - self.commands[:,:2]), dim=1)
        ang_vel_error = torch.square(base_ang_vel[:, 2] - self.commands[:, 2])
        height_error = torch.square((self.commands[:,3] - self.actor_root_states[:,2]))
        orient_error = torch.sum(torch.square(projected_gravity[:, :2]), dim=1)
        ang_vel_xy_error = torch.sum(torch.square(base_ang_vel[:, :2]), dim=1)
        joint_centerness = torch.sum(torch.square(self.dof_pos-(self.dof_limits_lower+self.dof_limits_upper)),dim=1)


        rew_lin_vel_xy = (1 - 10 * torch.sqrt(0.01 + lin_vel_error)) * self.rew_scales["lin_vel_xy_rew_scale"]
        rew_ang_vel_z = (1 - 10 * torch.sqrt(0.01 + ang_vel_error))* self.rew_scales["ang_vel_z_rew_scale"]
        rew_height = (-0.3 + 10* torch.sqrt(0.0009 + (height_error/0.1))) * self.costs["height_cost"]
        rew_orient = (-0.3 + 3 * torch.sqrt(0.01 + (orient_error/0.1))) * self.costs["orientation_cost"]
        rew_ang_vel_xy = (-0.36 + 6 * torch.sqrt(0.0036 + (ang_vel_xy_error/200))) * self.costs["ang_vel_xy_cost"]
        rew_joint_centerness = (0.6325 - 1 * torch.sqrt(0.4 + ((joint_centerness)/7.5))) * self.rew_scales["joint_central_reward"]


        coxa_contact = torch.norm(self.net_contact_forces[:, self.coxa_indices, :], dim=2) > 0
        femur_contact = torch.norm(self.net_contact_forces[:, self.femur_indices, :], dim=2) >0
        tibia_contact = torch.norm(self.net_contact_forces[:, self.tibia_indices, :], dim=2) >0
        feet_contact = torch.norm(self.net_contact_forces[:, self.feet_indices, :],dim=2) > 0

        rew_collision = (torch.sum(coxa_contact, dim=1) + torch.sum(femur_contact, dim=1) + torch.sum(tibia_contact, dim=1)) * self.costs["leg_collision_cost"] 
        rew_joint_acc = torch.sum(torch.square(self.last_dof_vel - self.dof_vel), dim=1) * self.costs["joint_acc"]



        

        # rew_torque = torch.sum(torch.square(self.torques_sensors*self.dof_vel*0.1),dim=1) * self.costs["torque_cost"] 
            
        # x = torch.square(base_lin_vel[:, 2])
        # rew_lin_vel_z = x * self.costs["lin_vel_z_cost"]

        feet_contact_indices = torch.nonzero(feet_contact)
        feet_not_contact_indices = torch.nonzero(~feet_contact)

        # dt_tensor   = torch.full((len(feet_contact_indices),1),self.dt, device=self.device)
        # zero_tensor = torch.full((len(feet_not_contact_indices),1),0.0, device=self.device)

        # self.feet_contact_times.index_put_(tuple(feet_contact_indices.T),dt_tensor, accumulate=True)
        # self.feet_contact_times.index_put_(tuple(feet_not_contact_indices.T),zero_tensor, accumulate=False)

        dt_tensor   = torch.full((len(feet_not_contact_indices),1),self.dt, device=self.device)
        zero_tensor = torch.full((len(feet_contact_indices),1),0.0, device=self.device)

        self.feet_not_contact_times.index_put_(tuple(feet_not_contact_indices.T),dt_tensor, accumulate=True)
        self.feet_not_contact_times.index_put_(tuple(feet_contact_indices.T),zero_tensor, accumulate=False)

        # max_feet_contact_time,_ = torch.max(self.feet_contact_times, dim=1)
        max_feet_air_time,_ = torch.max(self.feet_not_contact_times, dim=1)
        # print(self.feet_contact_times[0])
        # print(rew_feet_air_time[0])

        rew_feet_air_time = torch.pow(1.5 * max_feet_air_time.squeeze(), 4) * self.costs["feet_air_time_cost"]
        # rew_feet_air_time = torch.sum(torch.square(self.feet_not_contact_times.squeeze()), dim=1) * self.costs["feet_air_time_cost"]

        
        total_reward = rew_lin_vel_xy + rew_ang_vel_z + rew_height + rew_orient + rew_collision + rew_joint_acc + rew_ang_vel_xy + rew_feet_air_time
        total_reward = total_reward * self.dt
        time_out = torch.where(self.progress_buf > self.max_episode_length-1, 1, 0)
        reset = time_out
        # reset = time_out | torch.any(torch.norm(self.net_contact_forces[:, base_index, :], dim=1) > 0.0, dim=0)
        # reset = reset | torch.any(torch.norm(self.net_contact_forces[:, coxa_indices, :], dim=2) > 0.0, dim=1)
        # reset = reset | torch.any(torch.norm(self.net_contact_forces[:, femur_indices, :], dim=2) > 0.0, dim=1)
        # reset = reset | torch.any(torch.norm(self.net_contact_forces[:, tibia_indices, :], dim=2) > 0.0, dim=1)
        
     
        print(f"feet_air_time {rew_feet_air_time[0]}")
        print(f"lin_vel_xy {rew_lin_vel_xy[0]}")
        print(f"ang_vel_z {rew_ang_vel_z[0]}")
        # print(f"lin_vel_z {rew_ang_vel_xy[0]}")
        print(f"ang_vel_xy {rew_ang_vel_xy[0]}")

        print(f"height {rew_height[0]}")
        print(self.actor_root_states[0,2])
        # print(height_error[0])
        # print(f"orient {rew_orient[0]}")

        # print(f"sqrt: {torch.sqrt(1 + lin_vel_error)}")
        print(f"joint central{rew_joint_centerness[0]}")
        print(f"collision {rew_collision[0]}")
        # print(f"torque {rew_torque[0]}")
        print(f"joint_acc{rew_joint_acc[0]}")
        # print(f"proj_gravity{projected_gravity[0]}\n")
        # print(self.commands[0,:4])

        # print("/n")
        self.rew_buf[:], self.reset_buf[:] = total_reward.detach(), reset


        
    def compute_observation(self):
        self.gym.refresh_dof_state_tensor(self.sim)  # done in step
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        self.obs_buf[:] = compute_hexapod_observations(# tensors
                                                        self.actor_root_states,
                                                        self.commands,
                                                        self.dof_pos,
                                                        self.default_dof_pos,
                                                        self.dof_vel,
                                                        self.gravity_vec,
                                                        self.actions,
                                                        self.torques_sensors,
                                                        # scales
                                                        self.lin_vel_scale,
                                                        self.ang_vel_scale,
                                                        self.dof_pos_scale,
                                                        self.dof_vel_scale,
                                                        self.action_scale,
                                                        self.torque_scale
        )


@torch.jit.script
def compute_hexapod_observations(root_states,
                                 commands,
                                 dof_pos,
                                 default_dof_pos,
                                 dof_vel,
                                 gravity_vec,
                                 actions,
                                 torques,
                                 lin_vel_scale,
                                 ang_vel_scale,
                                 dof_pos_scale,
                                 dof_vel_scale,
                                 action_scale,
                                 torque_scale
                                ):

    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, float, float, float, float) -> Tensor
    
    base_quat = root_states[:,3:7]
    base_lin_vel = quat_rotate_inverse(base_quat, root_states[:, 7:10]) / lin_vel_scale
    base_ang_vel = quat_rotate_inverse(base_quat, root_states[:, 10:13]) / ang_vel_scale
    projected_gravity = quat_rotate(base_quat, gravity_vec)
    dof_pos_scaled = (dof_pos - default_dof_pos) / dof_pos_scale
    dof_vel_scaled = dof_vel / dof_vel_scale
    torques_scaled = torques / torque_scale

    commands_scaled = commands/torch.tensor([lin_vel_scale, lin_vel_scale, ang_vel_scale, 1.0], requires_grad=False, device=commands.device)

    obs = torch.cat((base_lin_vel,
                     base_ang_vel,
                     projected_gravity,
                     commands_scaled,
                     torques_scaled,
                     dof_pos_scaled,
                     dof_vel_scaled,
                     actions * action_scale
                     ), dim=1)
    # print(f"ACTIONS\n {actions}")
    return obs