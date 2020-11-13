```diff
@@ Warning: The code in this repository is under construction. For the best exprience, please, try it later (in few days). @@
```

# About
This repository contains code for the CoRL 2020 submission titled **Learning Object Manipulation Skills via Approximate State Estimation from Real Videos**.
In case of any question contact us at *vladimir.petrik@cvut.cz* or *makarand.tapaswi@inria.fr*.

Additional data:
[Supplementary materials](https://data.ciirc.cvut.cz/public/projects/2020Real2Sim/), 
[YouTube overview video](https://youtu.be/0bhO3KCKVa8), 
[Paper PDF](https://drive.google.com/file/d/1DuHan9oZXznDnXiCP7J6ogWn8FMAAkIJ/view).

Citation:
```bibtex
TBD.
```

# Code
The code is divided into a few main parts: 
- estimating/optimizing course states from a video,
- reinforcement learning of the control policies,
- benchmarking.

First part has its own installation instruction (separate conda env.), requires GPU, and uses [neural renderer](https://github.com/hiroharu-kato/neural_renderer).
RL and benchmarking parts share the same conda environment and use [rlpyt](https://github.com/astooke/rlpyt) for RL and [PyPhysX](https://github.com/petrikvladimir/pyphysx/) for simulation.
To simplify the experimentation, we provide you computed data for each step, so if you are interesting in RL part only you can download the extracted states as described below.

## Estimating states from video 

### Installation

### Required data

TBD: link to extracted segmentation masks, 

### Optimization

## Reinforcement learning of control policies

### Installation
```shell script
conda env create -f install_real2sim_rl.yml
conda activate real2sim_rl

export PYTHONPATH="${PYTHONPATH}:`pwd`"  #run in the repo root
```

### Required data
Either run states estimation from the previous section, or download and extract states from TBD into the folder *data/states/*.
Once this is done, you can visualize states using:
```shell script
python simulation/scripts/visualize_states.py data/states/1sA
```
which will open 3D viewer and shows states one by one. 
No physics simulation is performed in this visualization.
Example of visualization:

![](doc/policy_learning/visualize_states.gif) 

### Training the policy
The following command will train policy for action id 86 (*pull left to right*) on states *1sA*.
On standard 4 cores laptop without gpu it takes about 3 hours to finish. 
You can track learning progress in tensorboard.

```shell script
# add -cuda_id 0 if you have cuda installed
python policy_learning/manipulation_learning.py -log_dir data/policies -name exp_1sA_act_86 --linear_lr -states_folder data/states/1sA -angle_bound_scale 0.01 --without_object_obs -seed 0
```
See folder *policy_learning/scripts/* for a collection of bash scripts we used for training policies which performance was reported in the paper.
These bash scripts just call *manipulation_learning.py* script with different parameters to train policies e.g. with different domain randomization or on different states.

### Visualization of trained policy

You can either download learned polices from here TODO. Or use policy trained in the previous step.
In case of downloaded policies, just adjust experiment name.
```shell script
python policy_learning/manipulation_learning.py -log_dir data/policies -name exp_1sA_act_86 -states_folder data/states/1sA --without_object_obs --greedy_eval --render --realtime
```


## Benchmarking
For benchmarking, install and activate conda environment from RL section:
```
conda activate real2sim_rl
``` 

