
import time
import numpy as np

from trainer.rs import ReplaySchedulingTrainer

class MonteCarloTreeSearch(object):

    def __init__(self, node):
        """
        MonteCarloTreeSearchNode
        Parameters
        ----------
        node : mctspy.tree.nodes.MonteCarloTreeSearchNode
        """
        self.root = node

    def best_action(self, simulations_number=None, total_simulation_seconds=None):
        """

        Parameters
        ----------
        simulations_number : int
            number of simulations performed to get the best action

        total_simulation_seconds : float
            Amount of time the algorithm has to run. Specified in seconds

        Returns
        -------

        """

        if simulations_number is None :
            assert(total_simulation_seconds is not None)
            end_time = time.time() + total_simulation_seconds
            while time.time() < end_time:
                v = self._tree_policy()
                reward = v.rollout()
                v.backpropagate(reward)
        else :
            for _ in range(0, simulations_number):            
                v = self._tree_policy()
                reward = v.rollout()
                v.backpropagate(reward)
        # to select best child go for exploitation only
        return self.root.best_child(c_param=0.)

    def _tree_policy(self):
        """
        selects node to run rollout/playout for

        Returns
        -------

        """
        current_node = self.root
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node


class ReplaySchedulingMCTS(MonteCarloTreeSearch):

    def __init__(self, config, node, datasets): # can include more args here if needed
        super().__init__(node) # for inheriting methods and properties from parent
        self.config = config
        self.datasets = datasets

        # Create trainer
        trainer_fn = ReplaySchedulingTrainer
        if 'extension' in config['training'].keys():
            if config['training']['extension'] in ['agem', 'er']:
                from trainer.rs_extension import ReplaySchedulingTrainerExtension
                trainer_fn = ReplaySchedulingTrainerExtension
                #self.trainer = ReplaySchedulingTrainerExtension(config)
            elif config['training']['extension'] in ['coreset']:
                from trainer.rs_coreset_buffer import ReplaySchedulingTrainerCoreset
                trainer_fn = ReplaySchedulingTrainerCoreset
                
            elif config['training']['extension'] in ['der']:
                print('Using DER extension as trainer!')
                from trainer.rs_der import ReplaySchedulingTrainerDER
                trainer_fn = ReplaySchedulingTrainerDER
            elif config['training']['extension'] in ['derpp']:
                print('Using DER++ extension as trainer!')
                from trainer.rs_der import ReplaySchedulingTrainerDERPP
                trainer_fn = ReplaySchedulingTrainerDERPP
            elif config['training']['extension'] in ['mer']:
                print('Using MER extension as trainer!')
                from trainer.rs_mer import ReplaySchedulingTrainerMER
                trainer_fn = ReplaySchedulingTrainerMER
            elif config['training']['extension'] in ['hal']:
                print('Using HAL extension as trainer!')
                from trainer.rs_hal import ReplaySchedulingTrainerHAL
                trainer_fn = ReplaySchedulingTrainerHAL
        self.trainer = trainer_fn(config) 

    def run_search(self, simulations_number, c_param=0.1):

        verbose = self.config['session']['verbose']
        search_method = self.config['search']['method']

        res = {}
        rewards = []
        best_rewards = []
        rs = []
        accs = []
        best_accs = []
        t_elapsed = []
        best_reward = 0.0
        best_rs = None
        best_acc = None
        t0 = time.time()

        for i in range(0, simulations_number):      
            if search_method == 'mcts':      
                v = self._tree_policy(c_param)
                rollout_res = v.rollout()
            elif search_method == 'random':
                v = self.root
                rollout_res = v.rollout()

            # train CL network and get reward
            rs = rollout_res['rs']
            actions = rollout_res['actions']
            simulation_res = self.trainer.run_with_replay_schedule(self.datasets, 
                                                                replay_schedule=rs,
                                                                actions=actions,
                                                                rollout_id=i+1)
            reward = simulation_res['reward']
            # backpropagate reward
            v.backpropagate(reward)
            # Check best reward and store results
            if reward > best_reward:
                best_reward = reward
                best_rs = simulation_res['rs'].copy()
                best_acc = simulation_res['acc'] # test acc
            # Save results from iteration
            rewards.append(reward)
            best_rewards.append(best_reward)
            rs.append(simulation_res['rs'].copy())
            accs.append(simulation_res['acc'])
            best_accs.append(best_acc)
            t_elapsed.append(time.time() - t0) # in seconds

            print('\nIter {:d}/{:d} - Reward (ACC): {:.4f},'.format(i+1, simulations_number, reward))
            print('Replay Schedule: ')
            print(np.stack(simulation_res['rs'], axis=0))
            print('Test ACC: {:.4f}'.format(np.mean(simulation_res['acc'][-1, :])))
            if verbose > 0:
                acc = simulation_res['acc']
                if acc.shape[0] <= 10: # looks awful if printing for 20 tasks
                    print('Accuracies =')
                    for i in range(acc.shape[0]):
                        print('\t',end=',')
                        for j in range(acc.shape[1]):
                            print('{:5.4f}% '.format(acc[i,j]),end=',')
                        print()
            print()
            # Reset trainer
            self.trainer.reset()
        # return results as dict
        res = {}
        res['best_reward'] = best_reward
        res['best_rs'] = best_rs
        res['best_acc'] = best_acc
        res['rewards'] = rewards
        res['best_rewards'] = best_rewards
        res['accs'] = accs
        res['best_accs'] = best_accs
        res['rs'] = rs
        res['time_elapsed'] = t_elapsed
        return res


    def _tree_policy(self, c_param):
        """
        selects node to run rollout/playout for

        Returns
        -------

        """
        current_node = self.root
        while not current_node.is_terminal_node():
            #print('fully expanded: ', current_node.is_fully_expanded())
            if not current_node.is_fully_expanded():
                #print('fully expanded!')
                return current_node.expand()
            else:
                current_node = current_node.best_child(c_param)
                #print('after best child - current_node.is_terminal_node(): ', current_node.is_terminal_node())
        return current_node


    def get_results_from_replay_schedule(self, replay_schedule, actions):
        #torch.manual_seed(config['session']['seed'])
        #trainer = ReplaySchedulingTrainer(self.config)
        res = self.trainer.run_with_replay_schedule(self.datasets, replay_schedule, actions)
        return res