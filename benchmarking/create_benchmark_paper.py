#!/usr/bin/env python

# Copyright (c) CTU -- All Rights Reserved
# Created on: 2020-07-13
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#
# Create benchmark by sampling from parameters distribution.
# Current parameters are set for an easy benchmark. Modify minv, maxv values if you want to obtain hard benchmark samples.
# Already generated samples are stored in /data/benchmark_specification/benchmark.csv [hard] and benchmark1.csv [easy].

import pandas as pd
import numpy as np
from pyphysx_utils.transformations import quat_from_euler, multiply_transformations

from simulation.envs import GripperCylinderEnv
from simulation.utils.rotations import quat_from_azimuth_elevation

np.random.seed(0)

# hazi hele hposx hposy hposz o0posx o0posy o0posz (=o0hei/2) o0hei o0rad o1hei o1rad

minv = [
    np.deg2rad(-180),  # hazi
    np.deg2rad(0),  # hele
    -0.1,  # hposx
    -0.1,  # hposy
    0.1,  # hposz
    0.,  # o0posx
    0.,  # o0posy
    0.,  # o0posz
    0.06,  # o0hei
    0.04,  # o0rad
    0.06,  # o1hei
    0.04,  # o1rad
]
maxv = [
    np.deg2rad(180),  # hazi
    np.deg2rad(80),  # hele
    0.1,  # hposx
    0.0,  # hposy
    0.2,  # hposz
    0.,  # o0posx
    0.,  # o0posy
    0.,  # o0posz
    0.08,  # o0hei
    0.05,  # o0rad
    0.08,  # o1hei
    0.05,  # o1rad
]
data = np.random.uniform(minv, maxv, size=(10000, len(maxv)))
data[:, 7] = data[:, 8] / 2.

""" Find 1000 samples that are collision free. """
valid = np.zeros(data.shape[0]).astype(np.bool)
env = GripperCylinderEnv(two_objects=True)
for i in range(valid.shape[0]):
    # hand_pose, grip, obj_pose, env.obj_height, env.obj_radius = env.reset_state_sampler(env)
    hq = quat_from_azimuth_elevation(data[i, 0], data[i, 1])
    hpos = data[i, 2:5]
    obj_pose = (data[i, 5:8], (0, 0, 0, 1))

    env.obj_height = data[i, 8]
    env.obj_radius = data[i, 9]
    [env.obj.detach_shape(s) for s in env.obj.get_atached_shapes()]
    env.obj.attach_shape(env.create_cylinder(env.obj_height, env.obj_radius))
    env.reset_hand_pose((hpos, hq), env.FINGER_OPEN_POS)
    env.obj.set_global_pose(obj_pose)

    env.obj2_height = data[i, 10]
    env.obj2_radius = data[i, 11]
    [env.obj2.detach_shape(s) for s in env.obj2.get_atached_shapes()]
    env.obj2.attach_shape(env.create_cylinder(env.obj2_height, env.obj2_radius))
    env.obj.set_global_pose(obj_pose)
    o1pose = multiply_transformations((hpos, hq), (np.zeros(3), quat_from_euler('y', np.pi / 2)))
    env.obj2.set_global_pose(o1pose)

    valid[i] = not env.hand_plane_hit() and not env.hand_obj_hit() and not env.obj2_plane_hit()
    env.reset_hand_pose((hpos, hq), env.obj2_radius)
    valid[i] &= not env.hand_plane_hit() and not env.hand_obj_hit()

    print(sum(valid))
    if int(sum(valid)) == 1000:
        break

print(sum(valid))

df = pd.DataFrame(data=data[valid], columns=[
    'hazi', 'hele', 'hposx', 'hposy', 'hposz', 'o0posx', 'o0posy', 'o0posz', 'o0hei', 'o0rad', 'o1hei', 'o1rad'
])
df.to_csv('/tmp/benchmark1.csv')
