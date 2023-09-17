# Monte Carlo tree search for Replay Scheduling

This directory includes PyTorch code for reproducing the results in Section 4.1 **"Results on Replay Scheduling with Monte Carlo Tree Search"**.

* This directory is structured as follows:
```
./configs        : Directory with config files
./mcts           : Directory where MCTS code is placed
./trainer        : Directory where trainers for all methods are placed 
./training       : Directory where networks and helper code for training is placed
```

## Running Experiments
Run the experiments for all datasets within this directory. The config files for each experiment contains the hyperparameters that we used in the paper. All experiments are run for the 5 seeds used in the paper, i.e. SEED=1,2,3,4,5. 

* Split MNIST (5 tasks, 2 classes/task). Runtime: 5 seeds*1h20min
```
python run_mcts.py ./configs/rs_mcts/MNIST.yaml
```
* Split FashionMNIST (5 tasks, 2 classes/task) Runtime: 5 seeds*1h45min
```
python run_mcts.py ./configs/rs_mcts/FashionMNIST.yaml
```
* Split notMNIST (5 tasks, 2 classes/task) Runtime: 5 seeds*1h45min (Has to be preprocessed! See information below)
```
python run_mcts.py ./configs/rs_mcts/notMNIST.yaml
```
* Permuted MNIST (10 tasks, 10 classes/task) Runtime: 5 seeds*8.7h 
```
python run_mcts.py ./configs/rs_mcts/PermutedMNIST.yaml
```
* Split CIFAR-100 (20 tasks, 5 classes/task) Runtime: 5 seeds*7.3h
```
python run_mcts.py ./configs/rs_mcts/CIFAR100.yaml
```
* Split miniImagenet (20 tasks, 5 classes/task) Runtime: 5 seeds*7.8h (Has to be downloaded and preprocessed! See information below)
```
python run_mcts.py ./configs/rs_mcts/miniImagenet.yaml
```
**Be aware of the long runtimes**

The results from each experiment are saved in directory ```experiment/DATASET/rs_mcts/${SAMPLE_SELECTION}/M${MEMORY_SIZE}```.

### Running Baselines 
Experiments for the **ETS baseline** are executed similarly for all datasets:
```
python run_ets_baseline.py ./configs/ets/${DATASET}.yaml
```

Experiments for the **Heuristic baseline** are executed similarly for all datasets:
```
python run_heuristic_baseline.py ./configs/heuristic/${DATASET}.yaml
```

Experiments for the **Breadth-First Search (BFS)** are executed similarly (only for Split MNIST, FashionMNIST and notMNIST):
```
python run_bfs.py ./configs/bfs/${DATASET}.yaml
```

### Changing the Memory Parameters 

The memory size, memory examples per class, and memory selection method can be changed by changing the following parameters in the config files:
* **memory_limit** Memory size of replay memory
* **examples_per_class** Memory examples per class (disabled when config argument ```training/extension: coreset```) 
* **sample_selection** memory seelcteion method 
Find these parameters under ```replay``` in the config files:
```
...
replay:
  sample_selection: uniform / kmeans_features / kcenter_features / icarl (should be plain text or string)
  memory_limit: 10 (should be integer)
  examples_per_class: 5 (should be integer)
...
```

### Changing the Class Incremental Scenario (no task labels available) 

The continual learning setting can be changed by changing the following parameters in the config files:
* **scenario** The continual learning scenario, set to 'class'
* **extension** The extension of the trainer class, set to 'none' to obtain same number of samples/class in the memory
Find these arguments under ```training``` in the config files:
```
...
training:
  extension: none (should be plain text or string)
  scenario: class (should be plain text or string)
...
```


## Running Experiment for Applying Scheduling to Recent Replay Methods 
Experiments with replay scheduling combined with recent replay methods such as
Hindsight Anchor Learning (HAL), Meta Experience Replay (MER), Dark Experience Replay (DER),
and DER++ are executed using the config files in ```./configs/recent_replay_methods```. 
For running RS-MCTS combined with one of the replay methods, then execute:   
```
python run_mcts.py ./configs/recent_replay_methods/${METHOD}/${DATASET}.yaml
```
by setting the argument ${METHOD}="hal" or "mer" or "der" or "derpp".

The results will then be saved in ```./experiment/${DATASET}/${METHOD}/rs_mcts```.

To use the replay methods with ETS or Heuristic, change the run script in the command above to
```run_ets_baseline.py``` or ```run_heuristic_baseline.py```.

**Be aware of the long runtimes for RS-MCTS again**


## Running Tiny Memory Experiments (Efficiency of Replay Scheduling.)
Experiments in the tiny memory setting with RS-MCTS are executed similarly as above for all datasets:  
```
python run_mcts.py ./configs/tiny_memory_experiment/rs_mcts/${DATASET}.yaml
```
**Be aware of the long runtimes for RS-MCTS again**

Experiments for the baselines are executed similarly 
```
python run_baseline.py ./configs/tiny_memory_experiment/baselines/${BASELINE}/${DATASET}.yaml
```
where the argument ${BASELINE}="agem" or "er_ring" or "er_uniform".

The results from each experiment are saved in directory ```experiment_tiny_memory/${DATASET}/${METHOD}/M${MEMORY_SIZE}```.

## Running Single Task Replay Memory Experiment (Sec. 1 and Fig.1)
The experiment with the single task replay memory with task 1 data on Split MNIST can be run by executing: 
```
python run_exp_single_task_memory.py ./configs/single_task_replay_MNIST.yaml
```

The runtime for this experiment is arund 20 mins. Results from the experiment are saved in directory ```experiment/single_task_replay_MNIST```.


## Additional Setup Steps for Split notMNIST
A zip-folder of notMNIST dataset has been downloaded from the [Adversarial CL repository](https://github.com/facebookresearch/Adversarial-Continual-Learning). 

Some images could not be opened by the dataloaders, so we need to filter these images before running experiments. 

Execute the following command in the root directory:
```
python data/filter_notMNIST.py
```
The notMNIST dataset is then stored in ```src/mcts_single_cl_envs/datasets/notMNIST``` with subdirectories ```train/``` and ```test/``` 
with training and test images respectively. 

## Additional Setup Steps for Split miniImagenet
miniImagenet dataset was downloaded from [this Google drive repo with jpg files](https://drive.google.com/open?id=137M9jEv8nw0agovbUiEN_fPl_waJ2jIj), where we found the link in [this repo](https://github.com/yaoyao-liu/mini-imagenet-tools#about-mini-ImageNet). 

Download the directories ./train, ./val, and ./test from the Google drive. We need to split all classes into a training and a test set. Keep their subdirs in one directory stored in ./data directory and run the following command in the root directory:
```
python data/process_folder_miniimagenet.py
```
The splitted miniImagenet dataset is then stored in ```src/mcts_single_cl_envs/datasets/miniImagenet``` with subdirectories ```train/``` and ```test/``` 
with training and test images respectively.
