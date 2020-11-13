#!/usr/bin/env python

# Copyright (c) CTU -- All Rights Reserved
# Created on: 2020-06-4
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#
# Script for visualizing all states in a given trajectory

import argparse
import numpy as np

from pyrender.viewer import TextAlign
from simulation.utils.rotations import quat_from_azimuth_elevation
from simulation.utils.states import StateTrajectories, ActionClasses
from simulation.envs import GripperCylinderEnv
import quaternion as npq

parser = argparse.ArgumentParser()
parser.add_argument('states_dir', type=str, help='Folder in which states are stored.')
options = parser.parse_args()
print(options)

state_trajectories = StateTrajectories(options.states_dir)

env = GripperCylinderEnv(render=True, two_objects=not state_trajectories.trajectories[0].single_object())
env.reset()

label_action = env.renderer.add_label(' ', location=TextAlign.TOP_RIGHT, color='tab:blue')
label_video = env.renderer.add_label(' ', location=TextAlign.BOTTOM_RIGHT, color='tab:blue')
for action in ActionClasses:
    print('Processing action {}'.format(action.get_title()))
    for state_trajectory in state_trajectories.get_all_for_action(action):
        print(' VID: {} num frames: {}, time length: {:.2f}s'.format(
            state_trajectory.video_id, state_trajectory.num_frames, state_trajectory.time_length
        ))
        env.renderer.update_label_text(label_action, action.get_title())
        env.renderer.update_label_text(label_video, f'vid_id: {state_trajectory.video_id}')
        state_trajectory.transform_to_object_frame(xyz_scale=1.5)
        state_trajectory.compute_all_cache(np.arange(0., 10., 1 / 12))

        for i in range(state_trajectory.num_frames):
            if not env.renderer.is_active:
                exit(1)
            env.renderer.update()
            for _ in range(3):
                env.control_frequency.sleep()
            env.reset_hand_pose(
                (
                    state_trajectory.get_cached_value(['hposx', 'hposy', 'hposz'], iteration=i),
                    quat_from_azimuth_elevation(*state_trajectory.get_cached_value(['hazi', 'hele'], iteration=i))
                ),
                (1 - state_trajectory.get_cached_value('touch', iteration=i)) * env.FINGER_OPEN_POS
            )
            env.obj.set_global_pose(
                (
                    state_trajectory.get_cached_value(['o0posx', 'o0posy', 'o0posz'], iteration=i),
                    npq.from_rotation_vector(state_trajectory.get_cached_value(
                        ['o0rot0', 'o0rot1', 'o0rot2'], iteration=i
                    ))
                ),
            )
            if not action.is_single_object():
                env.obj2.set_global_pose(
                    (
                        state_trajectory.get_cached_value(['o1posx', 'o1posy', 'o1posz'], iteration=i),
                        npq.from_rotation_vector(state_trajectory.get_cached_value(
                            ['o1rot0', 'o1rot1', 'o1rot2'], iteration=i
                        ))
                    ),
                )
