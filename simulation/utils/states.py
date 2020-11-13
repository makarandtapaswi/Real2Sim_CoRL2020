#!/usr/bin/env python

# Copyright (c) CTU -- All Rights Reserved
# Created on: 2020-04-20
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
import pathlib
from enum import Enum
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import quaternion as npq
from pyphysx_utils.transformations import quat_from_euler, inverse_transform, multiply_transformations
from scipy.spatial.transform import Rotation


class ActionClasses(Enum):
    PULLING_LEFT_TO_RIGHT = 86
    PULLING_RIGHT_TO_LEFT = 87
    PUSHING_LEFT_TO_RIGHT = 93
    PUSHING_RIGHT_TO_LEFT = 94
    PICK_UP = 47
    PUT_BEHIND = 104
    PUT_IN_FRONT_OF = 105
    PUT_NEXT_TO = 107
    PUT_ONTO = 112

    def get_all_ids(self):
        if self is self.PULLING_LEFT_TO_RIGHT:
            return [8675, 13956, 6848, 20193, 10960, 15606]
        elif self is self.PULLING_RIGHT_TO_LEFT:
            return [13309, 3107, 11240, 2458, 10116, 11732]
        elif self is self.PUSHING_LEFT_TO_RIGHT:
            return [3694, 601, 9889, 4018, 6955, 6132]
        elif self is self.PUSHING_RIGHT_TO_LEFT:
            return [3949, 1987, 4218, 4378, 1040, 8844]
        elif self is self.PICK_UP:
            return [7194, 38559, 12359, 1838, 2875, 24925]
        elif self is self.PUT_BEHIND:
            return [1044, 6538, 3074, 12986, 779, 4388]
        elif self is self.PUT_IN_FRONT_OF:
            return [1663, 41177, 28603, 5967, 14841, 2114]
        elif self is self.PUT_NEXT_TO:
            return [4053, 19, 874, 3340, 1890, 1642]
        elif self is self.PUT_ONTO:
            return [8801, 7655, 13390, 757, 7504, 16310]

    def get_title(self):
        if self is self.PULLING_LEFT_TO_RIGHT:
            return 'pull left to right'
        elif self is self.PULLING_RIGHT_TO_LEFT:
            return 'pull right to left'
        elif self is self.PUSHING_LEFT_TO_RIGHT:
            return 'push left to right'
        elif self is self.PUSHING_RIGHT_TO_LEFT:
            return 'push right to left'
        elif self is self.PUT_BEHIND:
            return 'put behind'
        elif self is self.PUT_IN_FRONT_OF:
            return 'put in front of'
        elif self is self.PUT_NEXT_TO:
            return 'put next to'
        elif self is self.PUT_ONTO:
            return 'put onto'
        elif self is self.PICK_UP:
            return 'pick up'

    def contains_video(self, vid):
        return vid in self.get_all_ids()

    def is_single_object(self):
        """ Return true if trajectory contains only single object. """
        return (self is self.PULLING_LEFT_TO_RIGHT) or \
               (self is self.PULLING_RIGHT_TO_LEFT) or \
               (self is self.PUSHING_LEFT_TO_RIGHT) or \
               (self is self.PUSHING_RIGHT_TO_LEFT) or \
               (self is self.PICK_UP)


class StateTrajectory:
    """ Class that reads state trajectory extracted from the video. """

    def __init__(self, filename: str = None, data_frame: pd.DataFrame = None) -> None:
        if filename is not None:
            self.df = pd.read_csv(filename, skipinitialspace=True)
        else:
            self.df = data_frame
        self.video_id = int(str(filename).split('/')[-1].split('_')[0]) if filename is not None else 0
        self.camera_transformed_pos = None
        self.time_stamps = np.linspace(0., self.time_length, self.num_frames)
        self.cached_properties = {}

    @property
    def num_frames(self):
        """ Get number of frames extracted from the video. """
        return self.df.shape[0]

    @property
    def time_length(self):
        """ Get total length of the sequence extracted from video in seconds. """
        return self.num_frames / 12.

    @property
    def start_time(self):
        """ Start time of the sequence. By default zero, if not adjusted."""
        return self.time_stamps[0]

    @property
    def final_time(self):
        """ Final time of the sequence. By default equals to time_length(), if not adjusted."""
        return self.time_stamps[-1]

    def set_start_and_final_time(self, start_time, final_time):
        """ Set start and final time of the sequence. """
        self.time_stamps = np.linspace(start_time, final_time, self.num_frames)

    def compute_all_cache(self, timestamps_at_iteration):
        """ If you know you will access properties at particular timesteps, you can compute caches in advance instead of
        interpolating all the time. """
        for col in self.df.columns:
            self.cached_properties[col] = self.get_property_at_time(col, timestamps_at_iteration)

    def get_cached_value(self, prop, iteration):
        """ Get cached value for the given property. """
        if isinstance(prop, str):
            return self.cached_properties[prop][iteration]
        else:
            return [self.cached_properties[p][iteration] for p in prop]

    def set_first_touch_to_open(self):
        self.df.at[0, 'touch'] = 0

    def offset_last_hand_pos(self, pos):
        self.df.at[self.df.shape[0] - 1, 'hposx'] += pos[0]
        self.df.at[self.df.shape[0] - 1, 'hposy'] += pos[1]
        self.df.at[self.df.shape[0] - 1, 'hposz'] += pos[2]

    def single_object(self):
        """ Return true if states are defined for single object only. """
        return 'o1posx' not in self.df.columns

    def transform_to_object_frame(self, object_id=0, xyz_scale=1.):
        """
        First, compensate azimuth angle s.t. camera is always looking towards negative x-axis. It is achieved by
        computing objects and hand position in camera coordinate frame. The azimuth rotation of hand is adjusted by
        substraction.

        Second, all positions are updated s.t. obj is always located at [0,0] x,y position.

        Third, orientation of object is updated s.t. first rotation is always identity.
        """
        oid = 'o{}'.format(object_id)
        opos = self.df[[oid + 'posx', oid + 'posy', oid + 'posz']].to_numpy() * [1, 1, 0]
        hpos = self.df[['hposx', 'hposy', 'hposz']].to_numpy() * [1, 1, 0]

        yaw = self.df['cazi'].mean()
        t_cam = (self.get_camera_position() * [1, 1, 0], quat_from_euler('z', yaw))
        t_cam_inv = inverse_transform(t_cam)

        """ Positions in camera frame. """
        opos = npq.rotate_vectors(t_cam_inv[1], opos)
        hpos = npq.rotate_vectors(t_cam_inv[1], hpos)

        """ Offset zero object position """
        pos_offset = opos[0].copy() if self.single_object() else opos.copy()
        opos = opos - pos_offset
        hpos = hpos - pos_offset

        """ Compensate rotation of the object """
        orot = self.df[[oid + 'rot0', oid + 'rot1', oid + 'rot2']].to_numpy()
        orot0: npq.quaternion = npq.from_rotation_vector(orot[0])
        for rot in orot:
            rot[:] = npq.as_rotation_vector(orot0.inverse() * npq.from_rotation_vector(rot))

        self.camera_transformed_pos = -pos_offset
        self.df[oid + 'posx'] = xyz_scale * opos[:, 0]
        self.df[oid + 'posy'] = xyz_scale * opos[:, 1]
        for i in range(3):
            self.df[oid + 'rot{}'.format(i)] = orot[:, i]
        self.df['hposx'] = xyz_scale * hpos[:, 0]
        self.df['hposy'] = xyz_scale * hpos[:, 1]
        self.df['hazi'] = self.df['hazi'].to_numpy() - yaw
        self.df['cazi'] = 0
        self.df['cdis'] = 0
        self.df['cele'] = 0

        self.df[oid + 'posz'] = xyz_scale * self.df[oid + 'posz']
        self.df[oid + 'szz'] = xyz_scale * self.df[oid + 'szz']
        self.df['hposz'] = xyz_scale * self.df['hposz']

        if not self.single_object():
            oid = 'o1'
            orot = self.df[[oid + 'rot0', oid + 'rot1', oid + 'rot2']].to_numpy()

            opos = self.df[[oid + 'posx', oid + 'posy', oid + 'posz']].to_numpy() * [1, 1, 0]
            opos = npq.rotate_vectors(t_cam_inv[1], opos)
            opos = opos - pos_offset
            self.df[oid + 'posx'] = xyz_scale * opos[:, 0]
            self.df[oid + 'posy'] = xyz_scale * opos[:, 1]
            self.df[oid + 'posz'] = xyz_scale * self.df[oid + 'posz']
            self.df[oid + 'szz'] = xyz_scale * self.df[oid + 'szz']

            orot0: npq.quaternion = npq.from_rotation_vector(orot[0])
            for rot in orot:
                rot[:] = npq.as_rotation_vector(orot0.inverse() * npq.from_rotation_vector(rot))
            for i in range(3):
                self.df[oid + 'rot{}'.format(i)] = orot[:, i]

    def get_property_at_time(self, prop, t):
        """ Use linear interpolation to get given property at given time. """
        return np.interp(t, self.time_stamps, self.df[prop])

    def get_target_camera_position(self):
        cpos = self.camera_transformed_pos
        return np.array([0, 0, 0]) if cpos is None else cpos + [0., 1., 0.]

    def get_camera_position(self, delta_dis=0.):
        if self.camera_transformed_pos is not None:
            return self.camera_transformed_pos + [0, -delta_dis, 0]
        cdis, cazi, cele = self.df[['cdis', 'cazi', 'cele']].mean(0).to_numpy() + [delta_dis, 0, 0]
        return np.array([cdis * np.cos(cele) * np.sin(cazi), -cdis * np.cos(cele) * np.cos(cazi), cdis * np.sin(cele)])

    def plot2d_evolution(self, ax=None, gripper_size=0.04, object_size=None, camera=True,
                         gripper_color='tab:orange', object_color='tab:blue', object2_color='tab:green',
                         transparency_time=False, fig_setting=True):
        if ax is None:
            fig, ax = plt.subplots(1, 1, squeeze=True)  # type: Optional[plt.Figure], plt.Axes
        else:
            fig = None

        alphas = np.linspace(0.05 if transparency_time else 1., 1., self.df.shape[0])
        if gripper_color is not None:
            for i, v in enumerate(self.df[['hposx', 'hposy', 'hazi']].to_numpy()):
                self.plot_gripper2d(ax, *v, delta=gripper_size, color=gripper_color, alpha=alphas[i])
        if object_color is not None:
            o_prefix = 'o0'
            if object_size is None:
                for i, v in enumerate(
                        self.df[[o_prefix + 'posx', o_prefix + 'posy', o_prefix + 'szx', o_prefix + 'szy']].to_numpy()):
                    self.plot_box2d(ax, *v, color=object_color, alpha=alphas[i])
            else:
                for i, v in enumerate(self.df[[o_prefix + 'posx', o_prefix + 'posy']].to_numpy()):
                    self.plot_box2d(ax, *v, object_size, object_size, color=object_color, alpha=alphas[i])

        if object2_color is not None and 'o1posx' in self.df.columns:
            o_prefix = 'o1'
            if object_size is None:
                for i, v in enumerate(
                        self.df[[o_prefix + 'posx', o_prefix + 'posy', o_prefix + 'szx', o_prefix + 'szy']].to_numpy()):
                    self.plot_box2d(ax, *v, color=object2_color, alpha=alphas[i])
            else:
                for i, v in enumerate(self.df[[o_prefix + 'posx', o_prefix + 'posy']].to_numpy()):
                    self.plot_box2d(ax, *v, object_size, object_size, color=object2_color, alpha=alphas[i])

        if camera:
            cam = self.get_camera_position()
            ax.plot(cam[0], cam[1], marker=(3, 0, 60 + np.rad2deg(self.df['cazi'].mean())), markersize=20,
                    color='tab:green')
            cam1 = self.get_camera_position(delta_dis=0.08)
            ax.plot(cam1[0], cam1[1], marker=(4, 0, 45 + np.rad2deg(self.df['cazi'].mean())), markersize=20,
                    color='tab:green')
            viewaxis = np.concatenate([self.get_camera_position()[:2, np.newaxis],
                                       self.get_target_camera_position()[:2, np.newaxis]], axis=1)
            ax.plot(viewaxis[0, :], viewaxis[1, :], ':', color='tab:green')
        if fig_setting:
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            ax.set_title('States evolution')
            ax.axis('equal')
        return fig, ax

    @staticmethod
    def plot_gripper2d(ax, x, y, azimuth, delta=0.1, color='tab:orange', **plot_kwargs):
        rot = Rotation.from_euler('z', azimuth)
        lines = np.array([
            [[0, delta / 2, 0], [-delta, delta / 2, 0]],
            [[-delta, delta / 2, 0], [-delta, -delta / 2, 0]],
            [[-delta, -delta / 2, 0], [0, -delta / 2, 0]],
            [[-delta, 0, 0], [-delta * 2.5, 0, 0]]
        ])
        lines = rot.apply(lines.reshape(-1, 3)).reshape(lines.shape) + [x, y, 0]
        for line in lines:
            ax.plot(line[:, 0], line[:, 1], '-', color=color, **plot_kwargs)

    @staticmethod
    def plot_box2d(ax, x, y, lx, ly, color='tab:blue', **plot_kwargs):
        lines = np.array([
            [[-lx / 2, -ly / 2, 0], [lx / 2, -ly / 2, 0]],
            [[lx / 2, -ly / 2, 0], [lx / 2, ly / 2, 0]],
            [[lx / 2, ly / 2, 0], [-lx / 2, ly / 2, 0]],
            [[-lx / 2, ly / 2, 0], [-lx / 2, -ly / 2, 0]],
        ]) + [x, y, 0]
        for line in lines:
            ax.plot(line[:, 0], line[:, 1], '-', color=color, **plot_kwargs)

    def get_action(self):
        for action in ActionClasses:
            if action.contains_video(self.video_id):
                return action
        return None


class StateTrajectories:
    def __init__(self, folder='states/all') -> None:
        self.trajectories = []
        for p in pathlib.Path(folder).iterdir():
            if p.is_file():
                self.trajectories.append(StateTrajectory(filename=str(p)))

    def get_all_for_action(self, action_class):
        return [traj for traj in self.trajectories if action_class.contains_video(traj.video_id)]

    def get_from_video_id(self, video_id):
        video_id = int(video_id)
        return next((x for x in self.trajectories if x.video_id == video_id), [None])
