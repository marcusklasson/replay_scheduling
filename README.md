# Replay Scheduling in Continual Learning  

This repository includes code in PyTorch for the paper **"Learn the Time to Learn: Replay Scheduling in Continual Learning"**.

## Installation

* Create conda environment from file:
```
conda env create -f environment.yml
conda activate rs_in_cl
```
* The following structure is in the main directory:
```
/data                        : Directory for preprocessing datasets not provided by torchvision
/mcts_single_cl_envs         : Directory with code for MCTS experiments
/rl_policy_generalization    : Directory with code for DQN and A2C experiments
```

Entering the directories ```mcts_single_cl_envs/``` and ```rl_policy_generalization/``` to see code used in Section 4.1 and 4.2 respectively.
