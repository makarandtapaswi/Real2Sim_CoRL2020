#!/bin/bash

# ---  PULL/PUSH actions
for action in 86 87 93 94
do
  for seed in 0 1 2 3 4 5 6 7 8 9
  do
    for set in 1sA 1sB 1sC
    do
      python policy_learning/manipulation_learning.py -log_dir data/ -name ablation2_${set}_act_${action} --linear_lr -cuda_id 0 -states_folder states/${set} -angle_bound_scale 0.01 --without_object_obs -seed $seed
    done
  done
done

# --- PICK UP action train ---
for action in 47 # Action 47 is using dz=0
do
  for seed in 0 1 2 3 4 5 6 7 8 9
  do
    for set in 1sA 1sB 1sC
    do
      python policy_learning/manipulation_learning.py -log_dir data/ -name ablation2_${set}_act_${action} --linear_lr -cuda_id 0 -states_folder states/${set} -angle_bound_scale 0.01 --without_object_obs -seed $seed -dz 0.
    done
  done
done

# --- Two objects action train ---
for seed in 0 1 2 3 4 5 6 7 8 9
do
  for action in 104 105 107 112
  do
    for set in 2sA 2sB 2sC
    do
      python policy_learning/manipulation_learning.py -log_dir data/ -name ablation2_${set}_act_${action} --linear_lr -cuda_id 0 -states_folder states/${set} -angle_bound_scale 0.01 --without_object_obs -seed $seed
    done
  done
done
