# RL-based Policy Learning for Replay Scheduling

This repository includes PyTorch code for reproducing the results in Section 4.2 **"Policy Generalization to New Continual Learning Scenarios"**.


```
./configs                         : Config files for all methods
./dataloaders                     : Directory for creating datasets 
./envs                            : Directory with code for creating the CL environments
./rl                              : Directory with code for DQN and A2C
./trainer                         : Directory where trainers for all methods are placed 
./training                        : Directory where networks and helper code for training is placed
./transition_tables_new_dataset   : Directory with transition tables for each CL environment in New Dataset experiments
./transition_tables_shuffled      : Directory with transition tables for each CL environment in New Task Order experiments networks
```

## Running Experiments
Due to constraints of the zip.file size, we only include transition tables for Split MNIST in ```transition_tables_shuffled``` and 
Split notMNIST in ```transition_tables_new_dataset```.

The config files in ```./configs``` contain the hyperparameters that we used for the experiments. We run each experiments for the 3 seeds used in the paper, i.e. SEED=1,2,3. 

### New Task Orders
We provide config files with hyperparameters for DQN and A"C under ```configs/dqn/new_task_order``` and ```configs/a2c/new_task_order``` respectively. 
Run the experiments for each datasets by executing the following commands:

* Split MNIST (10 test environments Seed 10-19).
```
python run_dqn_multienv.py --config ./configs/dqn/new_task_orders/mnist.yaml

python run_a2c_multienv.py --config ./configs/a2c/new_task_orders/mnist.yaml
```

**Be aware of the long runtime**

The results from each experiment are saved in directory ```experiments/new_task_order/DATASET/ALGO```.

### New Dataset
We provide config files with hyperparameters for DQN and A"C under ```configs/dqn/new_dataset``` and ```configs/a2c/new_dataset``` respectively. 
Run the experiments for each datasets by executing the following commands:

* Split notMNIST (10 test environments Seed 0-9).
```
python run_dqn_multienv.py ./configs/dqn/new_dataset/notmnist.yaml

python run_a2c_multienv.py ./configs/a2c/new_dataset/notmnist.yaml
```

The results from each experiment are saved in directory ```experiments/new_dataset/DATASET/ALGO```.


### Running Baselines 
Experiments for the **ETS baseline** are executed similarly for all datasets:
```
python run_baseline.py --config ./configs/baseline_${DATASET}.yaml --baseline_policy ets 
```

Experiments for the **Random baseline** are executed similarly for all datasets:
```
python run_baseline.py --config ./configs/baseline_${DATASET}.yaml --baseline_policy random  
```

Experiments for the **Heuristic baselines** are executed similarly for all datasets:
```
python run_baseline.py --config ./configs/baseline_${DATASET}.yaml --baseline_policy ${HEURISTIC} --tau ${TAU}
```
by setting ```${HEURISTIC}``` to either one of ```heuristic1, heuristic2, heuristic3``` 
where ```Heur-GD=heuristic1```, ```Heur-LD=heuristic2```, and ```Heur-AT=heuristic3```.
Set the threshold parameter ```${TAU}``` as in Table 9 (see Appendix B.3) for the correspodning heuristic and dataset. 

The results from each experiment are saved in directory ```experiments/DATASET/BASELINE```.

Remember to set the number of environments to evaluate and the starting seed for the envs in the configs under the args:
```
n_envs: 
seed:
```

## Evaluation with provided results
We provide our generated results in directory ```./results``` for the New Task Order experiment on Split MNIST. 
Run command
```
python plot_results_with_rl_algos.py
```
to produce the table below which corresponds to the Average Rankings for Split MNIST in Table 4 (see Section 4.2). 

|         | Split MNIST |    - " -       |
|:-------:|:-----------:|:-----------:|
| Method  |     Rank    |     ACC    |
| Random  |     3.98    |     91.8 +/- 4.7    |
| ETS     |     3.82    |     91.8 +/- 5.0    |
| Heur-GD |     4.53    |     91.3 +/- 4.3    |
| Heur-LD |     4.67    |     91.0 +/- 4.1    |
| Heur-AT |     4.38    |     91.5 +/- 4.0    |
| DQN     |     3.46    |     93.0 +/- 2.7    |
| A2C     |     3.16    |     93.1 +/- 3.7    |


## Additional Setup Steps for Split notMNIST
A zip-folder of notMNIST dataset has been downloaded from the [Adversarial CL repository](https://github.com/facebookresearch/Adversarial-Continual-Learning). 

Some images could not be opened by the dataloaders, so we need to filter these images before running experiments. 

Execute the following command in the root directory:
```
python data/filter_notMNIST.py
```
The notMNIST dataset is then stored in ```src/rl_policy_generalization/datasets/notMNIST``` with subdirectories ```train/``` and ```test/``` 
with training and test images respectively. 

## Generating CL Environments with Breadth-First Search (BFS)
The script ```bfs_experiment.py``` was used for generating the CL environments with BFS for iterating through the whole tree-shaped action space.
Since running BFS takes around 9 hours for Split MNIST, we refer the reader to use the provided and already-generated CL environments that we used in our experiments in ```./transition_tables_shuffled``` and ```./transition_tables_new_dataset```.

We will provide the full details for generating the CL environments upon acceptance.
