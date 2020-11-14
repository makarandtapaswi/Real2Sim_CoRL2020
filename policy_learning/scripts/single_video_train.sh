#!/bin/bash

# 4 single video actions pull/push left/right
for video in 6848 8675 10960 13956 15606 20193 2458 3107 10116 11240 11732 13309 601 3694 4018 6132 6955 9889 1040 1987 3949 4218 4378 8844
do
  for set in 1sA 1sC
  do
    python policy_learning/manipulation_learning.py -log_dir data/policies -name single_video_${set}_vid_${video} --linear_lr -states_folder states/${set} -angle_bound_scale 0.01 --without_object_obs
  done
done

# pick up
for video in 1838 2875 7194 12359 24925 38559
do
  for set in 1sA 1sC
  do
    python policy_learning/manipulation_learning.py -log_dir data/policies -name single_video_${set}_vid_${video} --linear_lr -states_folder states/${set} -angle_bound_scale 0.01 --without_object_obs -dz 0.
  done
done

# put front/behind/next/onto
for video in 779 1044 3074 4388 6538 12986 1663 2114 5967 14841 41177 28603 19 874 1642 1890 3340 4053 757 7504 7655 8801 13390 16310
do
  for set in 2sA 2sC
  do
    python policy_learning/manipulation_learning.py -log_dir data/policies -name single_video_${set}_vid_${video} --linear_lr -states_folder states/${set} -angle_bound_scale 0.01 --without_object_obs
  done
done
