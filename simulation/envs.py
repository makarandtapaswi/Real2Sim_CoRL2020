#!/usr/bin/env python

# Copyright (c) CTU  - All Rights Reserved
# Created on: 3/18/20
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>


import os
from typing import List

import trimesh
import numpy as np
import pandas as pd
import quaternion as npq

from pyphysx import *
from pyphysx_utils.transformations import multiply_transformations, quat_from_euler
from pyphysx_utils.rate import Rate

from rlpyt.envs.base import Env, EnvStep, EnvInfo
from rlpyt.spaces.float_box import FloatBox

from simulation.utils.states import StateTrajectory
from simulation.utils.rotations import azimuth_elevation_from_quat, quat_from_azimuth_elevation


class GripperCylinderEnv(Env):
    FINGER_OPEN_POS = 0.075

    def __init__(self, horizon=100, render=False, realtime=False, reward_params=None, reset_state_sampler=None,
                 state_trajectories: List[StateTrajectory] = None, video_filename=None,
                 reset_on_singularity=True, reset_on_plane_hit=True, action_start_time=0., action_end_time=0.,
                 two_objects=False, dz=0.2, open_gripper_on_leave=True, close_gripper_on_leave=False,
                 use_max_reward=False, add_obstacle=False) -> None:
        super().__init__()
        self.use_max_reward = use_max_reward
        self.close_gripper_on_leave = close_gripper_on_leave
        self.open_gripper_on_leave = open_gripper_on_leave
        assert not (self.close_gripper_on_leave and self.open_gripper_on_leave)
        self.action_end_time = action_end_time
        self.action_start_time = action_start_time
        self.two_objects = two_objects
        self.reset_on_singularity = reset_on_singularity
        self.reset_on_plane_hit = reset_on_plane_hit
        self.realtime = realtime
        self.reset_state_sampler = reset_state_sampler
        self.reward_params = reward_params or GripperCylinderEnv.get_default_reward_params()
        self.state_trajectories = state_trajectories or []
        self.render = render
        self._horizon = horizon
        self.iter = 0
        self.num_sub_steps = 10

        self.control_frequency = Rate(12)

        """ Define action and observation space. """
        self._action_space = FloatBox(low=-1., high=1., shape=7)
        self._observation_space = self.get_obs(get_space=True)

        """ Create scene. """
        self.scene = Scene(scene_flags=[SceneFlag.ENABLE_FRICTION_EVERY_ITERATION, SceneFlag.ENABLE_STABILIZATION])
        self.plane_mat = Material(static_friction=0, dynamic_friction=0)
        self.obj_mat = Material(static_friction=1., dynamic_friction=1., restitution=0)
        self.plane_actor = RigidStatic.create_plane(material=self.plane_mat)
        self.scene.add_actor(self.plane_actor)
        self.hand_actors = self.create_hand_actors(use_mesh_fingers=False, use_mesh_base=False)
        self.joints = self.create_hand_joints()
        self.obj = self.create_obj()
        self.obj_height = 0.1
        self.obj_radius = 0.03
        if self.two_objects:
            self.obj2 = self.create_obj()
            self.obj2_height = 0.1
            self.obj2_radius = 0.03
        if add_obstacle:
            actor = RigidStatic()
            actor.attach_shape(Shape.create_box([0.02, 0.02, 0.2], self.obj_mat))
            actor.set_global_pose([0.075, 0.015, 0.1])
            self.scene.add_actor(actor)

        if self.render:
            from pyphysx_render.pyrender import PyPhysxViewer, RoboticTrackball
            import pyrender
            render_scene = pyrender.Scene()
            render_scene.bg_color = np.array([0.75] * 3)
            cam = pyrender.PerspectiveCamera(yfov=np.deg2rad(60), aspectRatio=1.414, znear=0.005)
            cam_pose = np.eye(4)
            cam_pose[:3, 3] = RoboticTrackball.spherical_to_cartesian(0.75, np.deg2rad(-90), np.deg2rad(60))
            cam_pose[:3, :3] = RoboticTrackball.look_at_rotation(eye=cam_pose[:3, 3], target=np.zeros(3), up=[0, 0, 1])
            nc = pyrender.Node(camera=cam, matrix=cam_pose)
            render_scene.add_node(nc)
            render_scene.main_camera_node = nc
            self.renderer = PyPhysxViewer(video_filename=video_filename, render_scene=render_scene,
                                          viewer_flags={
                                              'axes_scale': 0.2, 'plane_grid_spacing': 0.1,
                                          })
            self.renderer.add_physx_scene(self.scene)
            # from pyphysx_render.renderer import PyPhysXParallelRenderer
            # self.renderer = PyPhysXParallelRenderer(render_window_kwargs=dict(
            #     video_filename=video_filename, coordinates_scale=0.2, coordinate_lw=1.,
            #     cam_pos_distance=0.75, cam_pos_elevation=np.deg2rad(60), cam_pos_azimuth=np.deg2rad(-90.),
            # ))

        """ Compute trajectory caches. Stores quantities in MxN arrays [# of time steps, # of trajectories] """
        timesteps = np.arange(0., self.control_frequency.period() * (1 + self.horizon), self.control_frequency.period())
        self.demo_opos = np.zeros((len(timesteps), len(self.state_trajectories), 3))
        self.demo_hpos = np.zeros((len(timesteps), len(self.state_trajectories), 3))
        self.demo_grip = np.zeros((len(timesteps), len(self.state_trajectories)))
        self.demo_hrot = np.zeros((len(timesteps), len(self.state_trajectories), 2))

        if self.two_objects:
            self.demo_opos2 = np.zeros((len(timesteps), len(self.state_trajectories), 3))

        for i, st in enumerate(self.state_trajectories):
            half_height = 0.5 * st.get_property_at_time('o0szz', timesteps)
            self.demo_opos[:, i, 0] = st.get_property_at_time('o0posx', timesteps)
            self.demo_opos[:, i, 1] = st.get_property_at_time('o0posy', timesteps)
            self.demo_opos[:, i, 2] = np.maximum(st.get_property_at_time('o0posz', timesteps) - half_height, 0.)

            if self.two_objects:
                half_height2 = 0.5 * st.get_property_at_time('o1szz', timesteps)
                self.demo_opos2[:, i, 0] = st.get_property_at_time('o1posx', timesteps)
                self.demo_opos2[:, i, 1] = st.get_property_at_time('o1posy', timesteps)
                self.demo_opos2[:, i, 2] = np.maximum(st.get_property_at_time('o1posz', timesteps) - half_height2, 0.)

            self.demo_hpos[:, i, 0] = st.get_property_at_time('hposx', timesteps)
            self.demo_hpos[:, i, 1] = st.get_property_at_time('hposy', timesteps)
            self.demo_hpos[:, i, 2] = np.maximum(st.get_property_at_time('hposz', timesteps) - half_height, 0.)
            self.demo_grip[:, i] = (1 - st.get_property_at_time('touch', timesteps)) * self.FINGER_OPEN_POS
            self.demo_hrot[:, i, 0] = st.get_property_at_time('hazi', timesteps)
            self.demo_hrot[:, i, 1] = st.get_property_at_time('hele', timesteps)

        self.demo_hpos[int(action_end_time * 12):, :, :] = self.demo_opos[int(action_end_time * 12):, :, :].copy()
        if self.two_objects:
            self.demo_hpos[int(action_end_time * 12):, :, :] = self.demo_opos2[int(action_end_time * 12):, :, :].copy()

        self.demo_hpos[int(action_end_time * 12):, :, 2] += dz

        self.demo_grip_weak_closed = np.zeros((len(timesteps), len(self.state_trajectories)), dtype=np.bool)
        self.demo_grip_weak_closed[int(action_start_time * 12):int(action_end_time * 12)] = True  # those inside action
        self.demo_grip_weak_closed &= self.demo_grip < self.FINGER_OPEN_POS / 2  # that are also closed

        # self.demo_grip_strong_open = np.ones((len(timesteps), len(self.state_trajectories)), dtype=np.bool)
        # self.demo_grip_strong_open[int(action_start_time * 12):int(action_end_time * 12)] = False

        tmp_rot_vec = np.zeros((len(timesteps), len(self.state_trajectories), 3))
        for i, st in enumerate(self.state_trajectories):
            for j in range(3):
                tmp_rot_vec[:, i, j] = st.get_property_at_time('o0rot{}'.format(j), timesteps)
        self.demo_orot = npq.from_rotation_vector(tmp_rot_vec)

        if self.two_objects:
            tmp_rot_vec = np.zeros((len(timesteps), len(self.state_trajectories), 3))
            for i, st in enumerate(self.state_trajectories):
                for j in range(3):
                    tmp_rot_vec[:, i, j] = st.get_property_at_time('o1rot{}'.format(j), timesteps)
            self.demo_orot2 = npq.from_rotation_vector(tmp_rot_vec)

    def get_obs(self, get_space=False):
        """
            Get observation of state for RL. If get_space is true return space description.
            Space is defined as:
                0: time [1]
                1:7 hand pos + rotation vector [6]
                7: gripper state [1]
                8:14 obj pos + rotation vector [6],
                14: obj height
                15: obj radius
            In case of two objects:
                16:22 obj pos + rotation vector [6],
                22: obj height
                23: obj radius
        """
        if get_space:
            if self.two_objects:
                return FloatBox(
                    low=[0.] + [-1.] * 6 + [0.] + [-1.] * 6 + [0.] * 2 + [-1.] * 6 + [0.] * 2,
                    high=[1.] + [1.] * 6 + [self.FINGER_OPEN_POS] + [1.] * 6 + [0.2] * 2 + [1.] * 6 + [0.2] * 2
                )
            return FloatBox(
                low=[0.] + [-1.] * 6 + [0.] + [-1.] * 6 + [0.] * 2,
                high=[1.] + [1.] * 6 + [self.FINGER_OPEN_POS] + [1.] * 6 + [0.2] * 2
            )

        t = self.scene.simulation_time
        hand_pose = self.hand_actors[0].get_global_pose()
        grip = self.get_grip()
        obj_pose = self.obj.get_global_pose()
        if self.two_objects:
            obj2_pose = self.obj2.get_global_pose()
            return np.concatenate([
                [t / 10.],
                hand_pose[0], npq.as_rotation_vector(hand_pose[1]),
                [grip],
                obj_pose[0], npq.as_rotation_vector(obj_pose[1]),
                [self.obj_height, self.obj_radius],
                obj2_pose[0], npq.as_rotation_vector(obj2_pose[1]),
                [self.obj2_height, self.obj2_radius],
            ]).astype(np.float32)

        return np.concatenate([
            [t / 10.],
            hand_pose[0], npq.as_rotation_vector(hand_pose[1]),
            [grip],
            obj_pose[0], npq.as_rotation_vector(obj_pose[1]),
            [self.obj_height, self.obj_radius],
        ]).astype(np.float32)

    def step(self, action):
        self.iter += 1

        """ Step in pyphysx. """
        dt = self.control_frequency.period() / self.num_sub_steps
        dpose = (
            0.3 * np.tanh(action[:3]) * dt,
            quat_from_euler('xyz', np.pi * np.tanh(action[3:6]) * dt)
        )
        if self.open_gripper_on_leave and self.scene.simulation_time > self.action_end_time:
            self.set_grip(self.FINGER_OPEN_POS)
        elif self.close_gripper_on_leave and self.scene.simulation_time > self.action_end_time:
            self.set_grip(0.)
        else:
            grip = np.tanh(action[6]) * self.FINGER_OPEN_POS / 2 + self.FINGER_OPEN_POS / 2
            # grip = 0.04 if grip < 0.05 else self.FINGER_OPEN_POS
            self.set_grip(grip)
        pose = self.hand_actors[0].get_global_pose()
        for _ in range(self.num_sub_steps):
            pose = multiply_transformations(dpose, pose)
            self.hand_actors[0].set_kinematic_target(pose)
            self.scene.simulate(dt)

        if self.render:
            if self.iter == 1:
                self.renderer.clear_physx_scenes()
                self.renderer.add_physx_scene(self.scene)
            self.renderer.update()
            # self.renderer.render_scene(self.scene, recompute_actors=self.iter == 1)

        if self.realtime:
            self.control_frequency.sleep()

        """ Compute rewards """
        hp, hq = self.hand_actors[0].get_global_pose()
        op, oq = self.obj.get_global_pose()
        grip = self.get_grip()
        hrot = azimuth_elevation_from_quat(hq)

        hp[2] -= 0.5 * self.obj_height
        op[2] -= 0.5 * self.obj_height

        traj_rewards = np.zeros(self.demo_hpos.shape[1])
        b, s = self.reward_params['demo_hand_pos']['b'], self.reward_params['demo_hand_pos']['scale']
        traj_rewards += s * np.exp(-0.5 * np.sum((self.demo_hpos[self.iter] - hp) ** 2, axis=-1) * b)
        b, s = self.reward_params['demo_obj_pos']['b'], self.reward_params['demo_obj_pos']['scale']
        traj_rewards += s * np.exp(-0.5 * np.sum((self.demo_opos[self.iter] - op) ** 2, axis=-1) * b)
        b, s = self.reward_params['demo_hand_azi_ele']['b'], self.reward_params['demo_hand_azi_ele']['scale']
        traj_rewards += s * np.exp(-0.5 * np.sum((self.demo_hrot[self.iter] - hrot) ** 2, axis=-1) * b)
        b, s = self.reward_params['demo_obj_rot']['b'], self.reward_params['demo_obj_rot']['scale']
        traj_rewards += s * np.exp(-0.5 * npq.rotation_intrinsic_distance(self.demo_orot[self.iter], oq) * b)
        b, s = self.reward_params['demo_touch']['b'], self.reward_params['demo_touch']['scale']
        traj_rewards += s * self.demo_grip_weak_closed[self.iter] * np.exp(
            -0.5 * b * (grip - (self.obj_radius - 0.005)) ** 2)

        if self.two_objects:
            op2, oq2 = self.obj2.get_global_pose()
            op2[2] -= 0.5 * self.obj2_height
            b, s = self.reward_params['demo_obj_pos']['b'], self.reward_params['demo_obj_pos']['scale']
            traj_rewards += s * np.exp(-0.5 * np.sum((self.demo_opos2[self.iter] - op2) ** 2, axis=-1) * b)
            b, s = self.reward_params['demo_obj_rot']['b'], self.reward_params['demo_obj_rot']['scale']
            traj_rewards += s * np.exp(-0.5 * npq.rotation_intrinsic_distance(self.demo_orot2[self.iter], oq2) * b)
            b, s = self.reward_params['demo_touch']['b'], self.reward_params['demo_touch']['scale']
            traj_rewards += s * self.demo_grip_weak_closed[self.iter] * np.exp(
                -0.5 * b * (grip - (self.obj2_radius - 0.005)) ** 2)
        if traj_rewards.size == 0:
            reward = 0.
        else:
            reward = np.max(traj_rewards) if self.use_max_reward else np.mean(traj_rewards)

        """ Resolve singularities and collisions that we don't want. """
        if self.hand_plane_hit():
            if self.reset_on_plane_hit:
                return EnvStep(self.get_obs(), 0., True, EnvInfo())
            reward = 0.

        if np.abs(hrot[1]) > np.deg2rad(85):  # do not allow motion close to singularity
            if self.reset_on_singularity:
                return EnvStep(self.get_obs(), 0., True, EnvInfo())
            reward = 0.

        return EnvStep(self.get_obs(), reward / self.horizon, self.iter == self.horizon, EnvInfo())

    def reset(self):
        self.iter = 0
        if self.reset_state_sampler is None:
            self.reset_hand_pose(pose=((0, 0, 0.05), quat_from_euler('yz', [-np.pi / 2, -np.pi / 2])))
            self.obj.set_global_pose([0.0, 0.0, 0.05])
        else:
            while True:
                if self.two_objects:
                    hand_pose, grip, obj_pose, self.obj_height, self.obj_radius, obj2_pose, self.obj2_height, self.obj2_radius = self.reset_state_sampler(
                        self)
                    for s in self.obj2.get_atached_shapes():
                        self.obj2.detach_shape(s)
                    self.obj2.attach_shape(self.create_cylinder(self.obj2_height, self.obj2_radius))
                    self.obj2.set_global_pose(obj2_pose)
                else:
                    hand_pose, grip, obj_pose, self.obj_height, self.obj_radius = self.reset_state_sampler(self)
                [self.obj.detach_shape(s) for s in self.obj.get_atached_shapes()]
                self.obj.attach_shape(self.create_cylinder(self.obj_height, self.obj_radius))
                self.reset_hand_pose(hand_pose, grip)
                self.obj.set_global_pose(obj_pose)
                if not self.hand_plane_hit() and not self.hand_obj_hit():
                    break
                # else:
                #     print('reset collision detected: hand/plane: {},  hand/obj: {}'.format(self.hand_plane_hit(),
                #                                                                            self.hand_obj_hit()))
                #     print('Sample:', hand_pose, grip, obj_pose, self.obj_height, self.obj_radius)
        self.plane_mat.set_static_friction(np.random.uniform(0., 0.05))
        self.plane_mat.set_dynamic_friction(np.random.uniform(0., 0.05))
        self.scene.simulation_time = 0.
        return self.get_obs()

    def hand_plane_hit(self):
        for a in self.hand_actors:
            if a.overlaps(self.plane_actor):
                return True
        return False

    def hand_obj_hit(self):
        for a in self.hand_actors:
            if a.overlaps(self.obj):
                return True
        return False

    def obj2_plane_hit(self):
        if not self.two_objects:
            return False
        return self.obj2.overlaps(self.plane_actor)

    @staticmethod
    def randomized_reset_state_sampler(hazi_min=-np.deg2rad(360), hazi_max=np.deg2rad(360),
                                       hele_min=np.deg2rad(-80), hele_max=np.deg2rad(80),
                                       ohei_min=0.035, ohei_max=0.105, orad_min=0.025, orad_max=0.055,
                                       shared_sampler_dict=None):
        def sample(env: GripperCylinderEnv):
            hpos_std = shared_sampler_dict['hpos_std']
            sa = shared_sampler_dict['angle_bound_scale']
            q = quat_from_azimuth_elevation(np.random.uniform(hazi_min * sa, hazi_max * sa) + np.pi / 2,
                                            np.random.uniform(hele_min * sa, hele_max * sa))

            ohei = np.random.uniform(ohei_min, ohei_max)
            orad = np.random.uniform(orad_min, orad_max)
            tip_pos = np.zeros(3)
            tip_pos[:2] = np.random.normal(0., hpos_std, size=2)
            tip_pos[2] = ohei / 2. + np.random.normal(0., hpos_std, size=1)
            obj_pos = np.array([0., 0., ohei / 2.])
            obj_pos[:2] += np.random.normal(0., 1e-3, size=2)
            if not env.two_objects:
                return (tip_pos, q), GripperCylinderEnv.FINGER_OPEN_POS, (obj_pos, npq.one), ohei, orad

            tind = np.random.randint(0, env.demo_opos2.shape[1])
            tip_pos += env.demo_opos2[0, tind, :]
            o2pose = multiply_transformations((tip_pos, q), (np.zeros(3), quat_from_euler('y', np.pi / 2)))
            o2h = np.random.uniform(ohei_min, ohei_max)
            o2rad = np.random.uniform(orad_min, orad_max)
            return (tip_pos, q), o2rad, (obj_pos, npq.one), ohei, orad, o2pose, o2h, o2rad

        return sample

    @staticmethod
    def get_default_reward_params(fixed_obj_pose_scale=0., hand_obj_pos_dist_scale=0., demo_hand_pos=0.,
                                  demo_obj_pos=0., demo_hand_azi_ele=0., demo_touch=0., demo_obj_rot=0.):
        return {
            'fixed_obj_pose': {'b': 100., 'scale': fixed_obj_pose_scale, },
            'hand_obj_pos_dist': {'b': 100., 'scale': hand_obj_pos_dist_scale, },
            'demo_hand_pos': {'b': 100., 'scale': demo_hand_pos, },
            'demo_obj_pos': {'b': 100., 'scale': demo_obj_pos, },
            'demo_obj_rot': {'b': 10., 'scale': demo_obj_rot, },
            'demo_hand_azi_ele': {'b': 10., 'scale': demo_hand_azi_ele, },
            'demo_touch': {'b': 10000., 'scale': demo_touch, },
        }

    @property
    def horizon(self):
        return self._horizon

    @staticmethod
    def df_from_observations(observations):
        assert len(observations.shape) == 2
        assert observations.shape[1] == 16 or observations.shape[1] == 24
        t = observations[:, 0]
        hpos, hrot = observations[:, 1:4], observations[:, 4:7]
        grip = observations[:, 7]
        opos, orot = observations[:, 8:11], observations[:, 11:14]
        ohei, orad = observations[:, 14], observations[:, 15]

        helaz = npq.as_spherical_coords(npq.from_rotation_vector(hrot)) - (np.pi / 2, 0.)

        df = pd.DataFrame()
        df.insert(0, 'hposx', hpos[:, 0])
        df.insert(0, 'hposy', hpos[:, 1])
        df.insert(0, 'hposz', hpos[:, 2])
        df.insert(0, 'hazi', helaz[:, 1])
        df.insert(0, 'hele', helaz[:, 0])
        df.insert(0, 'touch', grip)
        df.insert(0, 'o0posx', opos[:, 0])
        df.insert(0, 'o0posy', opos[:, 1])
        df.insert(0, 'o0posz', opos[:, 2])
        df.insert(0, 'o0rot0', orot[:, 0])
        df.insert(0, 'o0rot1', orot[:, 1])
        df.insert(0, 'o0rot2', orot[:, 2])
        df.insert(0, 'ohei', ohei)
        df.insert(0, 'orad', orad)

        if observations.shape[1] > 16:
            o1pos, o1rot = observations[:, 16:19], observations[:, 19:22]
            o1hei, o1rad = observations[:, 22], observations[:, 23]
            df.insert(0, 'o1posx', o1pos[:, 0])
            df.insert(0, 'o1posy', o1pos[:, 1])
            df.insert(0, 'o1posz', o1pos[:, 2])
            df.insert(0, 'o1rot0', o1rot[:, 0])
            df.insert(0, 'o1rot1', o1rot[:, 1])
            df.insert(0, 'o1rot2', o1rot[:, 2])
            df.insert(0, 'o1hei', o1hei)
            df.insert(0, 'o1rad', o1rad)
        return df

    def create_cylinder(self, height=0.1, radius=0.03):
        cylinder = trimesh.primitives.Cylinder(height=height, radius=radius, sections=32)
        shape = Shape.create_convex_mesh_from_points(points=cylinder.vertices, material=self.obj_mat)
        return shape

    def create_obj(self):
        actor = RigidDynamic()
        actor.attach_shape(self.create_cylinder())
        actor.set_mass(1.)
        actor.set_global_pose([0, 0, 0.05])
        actor.set_angular_damping(0.05)
        actor.set_linear_damping(0.05)
        actor.set_max_linear_velocity(10.)
        actor.set_max_angular_velocity(2 * np.pi)
        self.scene.add_actor(actor)
        return actor

    def reset_hand_pose(self, pose, grip=None):
        if grip is None:
            grip = GripperCylinderEnv.FINGER_OPEN_POS
        self.hand_actors[0].set_global_pose(pose)
        self.hand_actors[1].set_global_pose(multiply_transformations(pose, [0, grip, 0.0584]))
        self.hand_actors[2].set_global_pose(multiply_transformations(pose, [0, -grip, 0.0584]))
        self.set_grip(grip)

    def get_grip(self):
        return self.joints[0].get_relative_transform()[0][1]
        # return self.joints[0].get_drive_position()[0][1]

    def set_grip(self, v):
        v = np.clip(v, 0., self.FINGER_OPEN_POS)
        self.joints[0].set_drive_position([0, v, 0])
        self.joints[1].set_drive_position([0, -v, 0])

    def create_hand_joints(self):
        j1 = D6Joint(self.hand_actors[0], self.hand_actors[1], local_pose0=[0, 0, 0.0584])
        j2 = D6Joint(self.hand_actors[0], self.hand_actors[2], local_pose0=[0, 0, 0.0584])
        for j in [j1, j2]:
            j.set_motion(D6Axis.Y, D6Motion.LIMITED)
            j.set_drive(D6Drive.Y, stiffness=1000., damping=200., force_limit=20.)
        j1.set_linear_limit(D6Axis.Y, 0., GripperCylinderEnv.FINGER_OPEN_POS)
        j2.set_linear_limit(D6Axis.Y, -GripperCylinderEnv.FINGER_OPEN_POS, 0.)
        return [j1, j2]

    def create_hand_actors(self, use_mesh_base=True, use_mesh_fingers=True):
        panda_dir = os.path.dirname(os.path.abspath(__file__)) + '/franka_panda/meshes/collision'
        hand_mat = Material(static_friction=1., dynamic_friction=1., restitution=0)

        trimesh_kwargs = dict(split_object=False, group_material=False)
        convex_kwargs = dict(material=hand_mat, quantized_count=64, vertex_limit=64)
        if use_mesh_base:
            mesh_hand: trimesh.Trimesh = trimesh.load('{}/hand.obj'.format(panda_dir), **trimesh_kwargs)
            hand_shape = Shape.create_convex_mesh_from_points(points=mesh_hand.vertices, **convex_kwargs)
        else:
            minp, maxp = np.array([-0.0316359, -0.10399, -0.0259248]), np.array([0.0316158, 0.100426, 0.0659622])
            hand_shape = Shape.create_box(size=maxp - minp, material=hand_mat)
            hand_shape.set_local_pose((minp + maxp) / 2)
        if use_mesh_fingers:
            mesh_finger: trimesh.Trimesh = trimesh.load('{}/finger.obj'.format(panda_dir), **trimesh_kwargs)
            finger1_shape = Shape.create_convex_mesh_from_points(points=mesh_finger.vertices, **convex_kwargs)
            finger2_shape = Shape.create_convex_mesh_from_points(points=mesh_finger.vertices, **convex_kwargs)
            finger2_shape.set_local_pose((np.zeros(3), quat_from_euler('z', np.pi)))
        else:
            finger1_shape = Shape.create_box(size=[0.02, 0.02, 0.12], material=hand_mat)
            finger2_shape = Shape.create_box(size=[0.02, 0.02, 0.12], material=hand_mat)
            finger1_shape.set_local_pose([0., 0.01, 0.07])
            finger2_shape.set_local_pose([0., -0.01, 0.07])

        for s in [hand_shape, finger1_shape, finger2_shape]:
            s.set_local_pose(multiply_transformations((0, 0, -0.15), s.get_local_pose()))

        actors = [RigidDynamic() for _ in range(3)]
        actors[0].attach_shape(hand_shape)
        actors[1].attach_shape(finger1_shape)
        actors[2].attach_shape(finger2_shape)
        actors[0].set_mass(1000 + 0.81)
        actors[1].set_mass(0.1 + 5.)
        actors[2].set_mass(0.1 + 5.)

        actors[0].set_rigid_body_flag(RigidBodyFlag.KINEMATIC, True)

        for a in actors:
            a.disable_gravity()
            a.set_angular_damping(0.5)
            a.set_linear_damping(0.5)
            self.scene.add_actor(a)
        return actors
