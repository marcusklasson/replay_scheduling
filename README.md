# Learn the Time to Learn: Replay Scheduling in Continual Learning  

[Marcus Klasson](https://marcusklasson.github.io/), [Hedvig Kjellström](https://www.kth.se/profile/hedvig), [Cheng Zhang](https://cheng-zhang.org/)

Accepted to Transactions on Machine Learning Research (TMLR) in 09/2023.

This repository includes Pytorch code for all experiments in the paper. 

Openreview: [https://openreview.net/forum?id=Q4aAITDgdP](https://openreview.net/forum?id=Q4aAITDgdP)

Arxiv: [https://arxiv.org/abs/2209.08660](https://arxiv.org/abs/2209.08660)



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

## Citation
If you find this work useful in your research, please cite our paper: 

```
@article{klasson2023learn,
    title={Learn the Time to Learn: Replay Scheduling in Continual Learning},
    author={Marcus Klasson and Hedvig Kjellstr{\"o}m and Cheng Zhang},
    journal={Transactions on Machine Learning Research},
    issn={2835-8856},
    year={2023},
    url={https://openreview.net/forum?id=Q4aAITDgdP},
    note={}
}
```
