#!/usr/bin/env python

# Copyright (c) CTU -- All Rights Reserved
# Created on: 2020-05-7
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import quaternion as npq

from simulation.utils.states import ActionClasses

parser = argparse.ArgumentParser()
parser.add_argument('benchmark_file', type=str)
parser.add_argument('trajectory_dir', type=str)
parser.add_argument('-performance_file', type=str, default='data/benchmark/performance.txt')
options = parser.parse_args()
print('Using options:', options)


def abs_angle_distance(angle1, angle2):
    diff = (angle1 - angle2 + np.pi) % (2 * np.pi) - np.pi
    diff[diff < -np.pi] += 2 * np.pi
    return diff


trajectory_dir = Path(options.trajectory_dir)
bench_data = pd.read_csv(options.benchmark_file)

""" Get action class from policy name. """
action_class = None
if 'vid' in options.trajectory_dir:
    video_id = int(options.trajectory_dir.split('_')[-1])
    for action in ActionClasses:
        if action.contains_video(video_id):
            action_class = action
            break
elif 'act' in options.trajectory_dir:
    action_id = int(options.trajectory_dir.split('_')[-1])
    action_class = ActionClasses(action_id)
else:
    raise NotImplementedError('Name must specify what to learn. Use vid_{id} or act_{id} at the end of your name')

action_id = int(action_class.value)

bench_data.insert(0, 'hand_ok', None)
bench_data.insert(0, 'obj_ok', None)
bench_data.insert(0, 'obj_rot_ok', 1)
bench_data.insert(0, 'hand_and_obj_ok', None)
bench_data.insert(0, 'oposx_last', None)
files = [p for p in trajectory_dir.iterdir()]
for p in tqdm(files):
    if not p.suffix == '.csv':
        continue
    tid = int(p.name.split('_')[-1].split('.')[0])
    traj = pd.read_csv(p)

    expected_azimuth = {
        ActionClasses.PULLING_LEFT_TO_RIGHT: np.deg2rad(180.),
        ActionClasses.PULLING_RIGHT_TO_LEFT: np.deg2rad(0.),
        ActionClasses.PUSHING_LEFT_TO_RIGHT: np.deg2rad(0.),
        ActionClasses.PUSHING_RIGHT_TO_LEFT: np.deg2rad(180.),
        ActionClasses.PICK_UP: None,
        ActionClasses.PUT_NEXT_TO: None,
        ActionClasses.PUT_IN_FRONT_OF: None,
        ActionClasses.PUT_BEHIND: None,
        ActionClasses.PUT_ONTO: None,
    }

    oposx = traj['o0posx'].to_numpy()
    obj_moving = np.abs(np.diff(oposx, append=oposx[-1])) * 12 > 1e-3

    exp_az_val = expected_azimuth[action_class]
    if exp_az_val is not None:
        ok_hazi = abs_angle_distance(traj['hazi'].to_numpy()[obj_moving], exp_az_val) < np.deg2rad(90.)
        ok_hele = (np.abs(traj['hele'].to_numpy()[obj_moving]) < np.deg2rad(80.)).all()
        bench_data.at[tid, 'hand_ok'] = float(np.sum(ok_hazi) > 0.75 * ok_hazi.shape[0] and ok_hele)
    else:
        bench_data.at[tid, 'hand_ok'] = float(True)

    o0rot = traj[['o0rot0', 'o0rot1', 'o0rot2']].tail(1).to_numpy()
    q = npq.from_rotation_vector(o0rot)
    v = npq.rotate_vectors(q, [0, 0, 1])
    if action_class in [ActionClasses.PULLING_LEFT_TO_RIGHT, ActionClasses.PUSHING_LEFT_TO_RIGHT]:
        bench_data.at[tid, 'obj_ok'] = float(traj['o0posx'].tail(1).to_numpy() > 0.05 and v[0, 2] > 0.5)
        bench_data.at[tid, 'obj_rot_ok'] = float(v[0, 2] > 0.5)
    elif action_class in [ActionClasses.PULLING_RIGHT_TO_LEFT, ActionClasses.PUSHING_RIGHT_TO_LEFT]:
        bench_data.at[tid, 'obj_ok'] = float(traj['o0posx'].tail(1).to_numpy() < -0.05 and v[0, 2] > 0.5)
        bench_data.at[tid, 'obj_rot_ok'] = float(v[0, 2] > 0.5)
    elif action_class in [ActionClasses.PICK_UP]:
        obottom = traj['o0posz'].tail(1).to_numpy() - traj['ohei'].tail(1).to_numpy() / 2
        bench_data.at[tid, 'obj_ok'] = float(obottom > 0.01)
    elif action_class in [ActionClasses.PUT_BEHIND, ActionClasses.PUT_IN_FRONT_OF, ActionClasses.PUT_NEXT_TO]:
        x0 = traj['o0posx'].tail(1).to_numpy()
        y0 = traj['o0posy'].tail(1).to_numpy()
        d0 = np.sqrt(x0 ** 2 + y0 ** 2)  # d0 is close to zero, i.e. o0 does not move
        x = traj['o1posx'].tail(1).to_numpy()
        y = traj['o1posy'].tail(1).to_numpy()
        d = np.sqrt(x ** 2 + y ** 2)  # d is in a given radius 50cm
        z = traj['o1posz'].tail(1).to_numpy()  # z < 5cm; i.e. object is on the table (maximum height is 5cm)
        th = np.arctan2(y, x)  # angle to x-axis [-pi, pi]

        expected_th_mean = {
            ActionClasses.PUT_BEHIND: [np.deg2rad(90.)],
            ActionClasses.PUT_IN_FRONT_OF: [np.deg2rad(-90)],
            ActionClasses.PUT_NEXT_TO: [np.deg2rad(0), np.deg2rad(180), np.deg2rad(-180)]
        }
        oth_ok = False
        for exp_th in expected_th_mean[action_class]:
            if exp_th - np.deg2rad(60) < th < exp_th + np.deg2rad(60):
                oth_ok = True
                break
        bench_data.at[tid, 'obj_ok'] = float(oth_ok and z < 0.05 and d < 0.5 and d0 < 1e-2)

    elif action_class in [ActionClasses.PUT_ONTO]:
        x0 = traj['o0posx'].tail(1).to_numpy()
        y0 = traj['o0posy'].tail(1).to_numpy()
        d0 = np.sqrt(x0 ** 2 + y0 ** 2)  # d0 is close to zero, i.e. o0 does not move
        x = traj['o1posx'].tail(1).to_numpy()
        y = traj['o1posy'].tail(1).to_numpy()
        d = np.sqrt(x ** 2 + y ** 2)  # d is close to zero (given by radius of o0)
        z = traj['o1posz'].tail(1).to_numpy()
        h0 = traj['ohei'].tail(1).to_numpy()
        r0 = traj['orad'].tail(1).to_numpy()
        h1 = traj['o1hei'].tail(1).to_numpy()
        # bottom of manipulated object is above height of the zero object (with eps.)
        # center of manipulated object is in radius of o0
        # o0 does not move
        bench_data.at[tid, 'obj_ok'] = float(z - h1 / 2 + 1e-3 > h0 and d < r0 and d0 < 1e-2)

    bench_data.at[tid, 'hand_and_obj_ok'] = float(bench_data.at[tid, 'hand_ok'] + bench_data.at[tid, 'obj_ok'] > 1.99)
    bench_data.at[tid, 'oposx_last'] = float(traj['o0posx'].tail(1).to_numpy())

ho = np.sum(bench_data['hand_ok'].to_numpy()) / bench_data.shape[0]
oo = np.sum(bench_data['obj_ok'].to_numpy()) / bench_data.shape[0]
oroto = np.sum(bench_data['obj_rot_ok'].to_numpy()) / bench_data.shape[0]
hoo = np.sum(bench_data['hand_and_obj_ok'].to_numpy()) / bench_data.shape[0]

performance_file = Path(options.performance_file)

if not performance_file.exists():
    file_object = open(performance_file, 'w')
    file_object.write('policy_id,hand_ok,obj_ok,obj_rot_ok,hand_and_obj_ok\n')
    file_object.close()

file_object = open(performance_file, 'a')
file_object.write('{},{},{},{},{}\n'.format(options.trajectory_dir.split('/')[-1], ho, oo, oroto, hoo))
file_object.close()

import matplotlib.pyplot as plt

cols = ['hazi', 'hele', 'hposx', 'hposy', 'hposz', 'o0posz', 'o0hei', 'o0rad', 'o1hei', 'o1rad']
fig_ratio, axes_ratio = plt.subplots(5, 2, squeeze=False, sharex=False, sharey=True,
                                     figsize=(6.4 * 2, 4.8 * 2))
fig, axes = plt.subplots(5, 2, squeeze=False, sharex=False, sharey=True,
                         figsize=(6.4 * 2, 4.8 * 2))
axes = axes.ravel()
axes_ratio = axes_ratio.ravel()
for i, col in enumerate(cols):
    axes[i].clear()
    df_pos = bench_data.loc[bench_data['hand_and_obj_ok'] > 0.99]
    df_neg = bench_data.loc[bench_data['hand_and_obj_ok'] < 0.01]
    n, bins, _ = axes[i].hist(x=[df_pos[col].to_numpy(), df_neg[col].to_numpy()], bins=20, histtype='bar', alpha=0.5,
                              rwidth=0.85,
                              color=['tab:green', 'tab:red'])
    # n [2x20]
    axes_ratio[i].bar(bins[:-1] + (bins[1:] - bins[:-1]) / 2, n[0, :] / np.sum(n, axis=0), alpha=0.5,
                      width=(bins[-1] - bins[0]) / 20 * 0.85, color='tab:blue')
    axes[i].set_ylabel('{}'.format(col))
    axes_ratio[i].set_ylabel('{}'.format(col))
    axes_ratio[i].set_ylim(0, 1)

fig.suptitle('{}; Number of valid (green) / Number of invalid (red); Success rate: {}'.format(
    options.trajectory_dir.split('/')[-1], hoo))
fig_ratio.suptitle('{}; Ratio of success/total; Success rate: {}'.format(
    options.trajectory_dir.split('/')[-1], hoo))

fig.savefig('{}/../success_marginal_{}.png'.format(options.trajectory_dir, options.trajectory_dir.split('/')[-1]))
fig_ratio.savefig('{}/../ratio_marginal_{}.png'.format(options.trajectory_dir, options.trajectory_dir.split('/')[-1]))
