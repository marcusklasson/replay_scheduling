
import os, argparse, time
import numpy as np
from omegaconf import OmegaConf

import torch

from training.utils import print_log_acc_bwt, set_random_seed, save_pickle, load_pickle
from training.vis_utils import plot_task_accuracy, plot_replay_schedule_bubble_plot
from envs.wrappers import MultiEnv, MultiEnvNew
from envs.action_spaces import DiscreteActionSpace
from envs.utils import *
from envs.env_fixed_seed import EnvFixedSeed
from envs.env_table import EnvTable, EnvTableFromFile

from rl.dqn.dqn import DQN 
from rl.dqn.model import DQNPolicy 
from rl.utils import test_agent, test_agent_in_env

t_start = time.time()

# Arguments
parser = argparse.ArgumentParser(description='Coding...')
# Load the config file
parser.add_argument('--config', type=str, default='./configs/dqn/new_task_orders/mnist.yml')
flags =  parser.parse_args()
args = OmegaConf.load(flags.config)

print()

########################################################################################################################

def train_episode(args, i_episode, agent, multi_env):
    # Shorthands
    device = args.device 
    n_tasks = args.n_tasks
    batch_size = args.dqn.batch_size
    n_batch_updates = args.dqn.n_batch_updates

    # make some empty lists for logging.
    ep_obs = []          # for observations
    ep_acts = []         # for actions
    ep_loss = []         # for measuring huber loss
    ep_td_errors = []    # for measuring td errors
    ep_rews = []            # list for rewards accrued throughout ep
    ep_q = []               # for measuring average Q-values
    ep_q_target = []
    ep_accs = []
    ep_rs = []
    ep_grad_abs_max, ep_grad_mean_sq = [], []

    # Get initial observation from starting distribution
    obs_t, infos = multi_env.reset()
    #print('reset: ', obs_t)
    ep_obs.append(obs_t)
    ep_accs.append([info['acc'] for info in infos])

    # This is a rollout
    for t in range(1, n_tasks):
        acts = []
        for obs in obs_t:
            act = agent.predict(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0),
                    torch.tensor([t-1], dtype=torch.long, device=device),
                    deterministic=False)
            acts.append(act.item())

        obs_tp1, rews, dones, infos = multi_env.step(t, acts)

        # save action, reward, and info
        ep_acts.append(acts)
        ep_rews.append(rews)
        ep_accs.append([info['acc'] for info in infos])
        ep_rs.append([info['task_proportion'] for info in infos])

        # Store the transition in memory if num_actions(task) > 1
        td_errors, losses, q_vals, q_targets, grad_abs_max, grad_mean_sq = [], [], [], [], [], [] 
        for ob, act, rew, ob_tp1, done in zip(obs_t, acts, rews, obs_tp1, dones):
            agent.replay_buffer.add(ob, act, rew, ob_tp1, float(done), int(t-1))

            if agent.learning_started: # moved this inside storing transitions loop
                # batch update will be performed for every new stored sample, so 1 update per env
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                td_errors_i, losses_i, q_vals_i, q_targets_i, grad_abs_max_i, grad_mean_sq_i = [], [], [], [], [], [] 
                for _ in range(n_batch_updates):
                    batch_summary = agent.update(batch_size)
                    # Store results from batch update
                    td_errors_i.append(batch_summary['td_error'])
                    losses_i.append(batch_summary['loss'])
                    q_vals_i.append(batch_summary['q_selected'])
                    q_targets_i.append(batch_summary['q_selected_target'])

                td_errors.append(td_errors_i)
                losses.append(losses_i)
                q_vals.append(q_vals_i)
                q_targets.append(q_targets_i)
                # update target network
                if ((agent.n_param_updates) % agent.target_update_freq == 0):
                    agent.hard_target_update()

        agent.n_timesteps += 1 # used for updating target network
        obs_t = obs_tp1
        ep_obs.append(obs_t)

        if agent.learning_started:
            ep_td_errors.append(td_errors)
            ep_loss.append(losses)
            ep_q.append(q_vals)
            ep_q_target.append(q_targets)
            
        
    # Update logger with scalars
    if agent.learning_started and (agent.n_param_updates % args.log_freq == 0):# and ((i_episode+1) >= args.log_burn_in):
        steps_done = agent.n_param_updates
        agent.log_statistics(category='q_value', data=np.array(ep_q), it=steps_done)
        agent.log_statistics(category='q_value_target', data=np.array(ep_q_target), it=steps_done)
        agent.log_statistics(category='loss', data=np.array(ep_loss), it=steps_done)
        agent.log_statistics(category='td_error', data=np.array(ep_td_errors), it=steps_done)
        # gradient statistics
        for name, stats in batch_summary['grad_stats'].items():
            agent.log_scalar(category='grad_abs_max', key='{:s}'.format(name), 
                            value=stats['abs_max'], it=steps_done)
            agent.log_scalar(category='grad_mean_sq', key='{:s}'.format(name), 
                            value=stats['mean_sq_value'], it=steps_done)
    # after episode
    if agent.learning_started:
        agent.after_episode()
    # Store information about episode
    ep_info = {}
    ep_info['obs'] = ep_obs 
    ep_info['acts'] = ep_acts 
    ep_info['rews'] = ep_rews
    ep_info['loss'] = ep_loss
    ep_info['td_errors'] = ep_td_errors
    ep_info['q'] = ep_q
    ep_info['q_target'] = ep_q_target
    ep_info['rs'] = ep_rs
    ep_info['accs'] = ep_accs
    return ep_info


def train(args, agent, train_envs, valid_envs):
    """ Run Q-learning algorithm in continual learning scenario.
    """
    # Create shorthands
    n_tasks = args.n_tasks
    n_episodes = args.n_episodes
    batch_size = args.dqn.batch_size
    buffer_size = args.buffer_size

    val_seeds = [env.seed for env in valid_envs.envs]
    log_res = {seed: [] for seed in val_seeds}
    
    accs = {'train': [[]]*len(train_envs), 
            'valid': [[]]*len(valid_envs),
            }

    print('Start training...')
    print()
    for i_episode in range(n_episodes):

        # Check if learning has started
        if (len(agent.replay_buffer) >= batch_size) and args.start_learning=='batch_size_full':
            agent.learning_started = True
        elif (len(agent.replay_buffer) >= buffer_size) and args.start_learning=='buffer_full':
            agent.learning_started = True
        elif (len(agent.replay_buffer) >= args.start_learning_min_random_exp) and args.start_learning=='min_random_exp':
            agent.learning_started = True

        # run episode in envs
        t0 = time.time()
        ep_info = train_episode(args, i_episode, agent, train_envs)

        # Logging per environment
        for k in range(len(train_envs)):
            env = train_envs.envs[k]
            ep_return = np.sum([rew[k] for rew in ep_info['rews']])
            acc = np.mean(ep_info['accs'][-1][k][:n_tasks])
            env.log_scalar('return', 'per_episode', ep_return, i_episode+1)
            env.log_scalar('ACC', 'per_episode', acc, i_episode+1)

        # Logging for agent
        agent.log_scalar('hyper_params', 'epsilon', agent.exploration_rate, i_episode+1)
        agent.log_scalar('hyper_params', 'buffer_size', len(agent.replay_buffer), i_episode+1)
        if agent.prioritized_er and agent.beta is not None:
            agent.log_scalar('hyper_params', 'beta', agent.beta, i_episode+1)
        # Print and log results
        if (i_episode+1) % args.print_freq == 0:
            avg_loss = np.mean(ep_info['loss']) if len(ep_info['loss']) > 0 else 0.
            avg_td_error = np.mean(ep_info['td_errors']) if len(ep_info['td_errors']) > 0 else 0.
            avg_q = np.mean(ep_info['q']) if len(ep_info['q']) > 0 else 0.
            avg_q_target = np.mean(ep_info['q_target']) if len(ep_info['q_target']) > 0 else 0.

            # Log time
            agent.log_scalar('time', 'per_episode', time.time()-t0, i_episode+1)
            agent.log_scalar('time', 'wall_clock', (time.time()-t_start)/60, i_episode+1)
            # Print episode info
            print('[Episode {}/{}] s/episode: {:.3f}, total time (min): {:1f}'.format(i_episode+1, 
                                                                                    n_episodes, 
                                                                                    time.time()-t0, 
                                                                                    (time.time()-t_start)/60))
            print('Avg. Loss: {:.3e}'.format(avg_loss))
            if (args.verbose > 0) and agent.learning_started:
                print('Avg. TD Error: {:.3e}'.format(avg_td_error))
                print('Avg. Q-value: {:.3f}'.format(avg_q))
                print('Avg. Q-Target: {:.3f}'.format(avg_q_target))
            print('Epsilon: {:.3f}'.format(agent.exploration_rate))
            if agent.prioritized_er and agent.beta is not None:
                print('Beta: {:.3f}'.format(agent.beta))
            #print()

            for k in range(len(train_envs)):
                ep_return = np.sum([rew[k] for rew in ep_info['rews']])
                acc = np.mean(ep_info['accs'][-1][k][:n_tasks])
                accs = np.array([ep_info['accs'][i][k][:n_tasks] for i in range(n_tasks)])
                assert acc == np.mean(accs[-1])
                acc_full = np.sum(accs) / np.sum(accs > 0).astype(np.float32)
                actions = [acts[k] for acts in ep_info['acts']]
                print('Environment {:d}: Return: {:.5f}, ACC: {:.5f}, ACC (full): {:.5f}, Actions: {}'.format(k+1, 
                        ep_return, acc, acc_full, actions))
            print()

        # Evaluate policy network 
        if agent.learning_started and ((i_episode+1) % args.eval_freq==0):
            if not os.path.exists(args.log_dir + '/eval'):
                os.makedirs(args.log_dir + '/eval', exist_ok=True)

            if (i_episode+1) % args.print_freq == 0:
                print('[Validation episode {:d}]'.format(i_episode+1))
            for k, env in enumerate(valid_envs.envs):
                t0 = time.time()
                res = test_agent_in_env(args, agent, env, split='test') #test(args, agent, env, split='test') # can evaluate return on test set for validation env 
                ret = res['return']
                acc = np.mean(res['acc'][-1, :n_tasks])
                accs = res['acc'][:, :n_tasks]
                acc_full = np.sum(accs) / np.sum(accs > 0).astype(np.float32)
                agent.log_scalar('seed%d' %(env.seed), 'return', ret, i_episode+1)
                agent.log_scalar('seed%d' %(env.seed), 'ACC', acc, i_episode+1)
                for t, action in enumerate(res['act'][1:], start=2):
                    agent.log_scalar('seed%d' %(env.seed), 'actions/task%d'%(t+1), action, i_episode+1)

                plot_task_accuracy(accs=res['acc'], 
                                    save_dir=args.log_dir + '/eval', 
                                    fname='task_acc_seed%d' %(env.seed))
                if args.n_tasks == 5:
                    plot_replay_schedule_bubble_plot(replay_schedule=res['rs'], 
                                        save_dir=args.log_dir + '/eval', 
                                        fname='task_prop_seed%d' %(env.seed))
                log_res[env.seed].append((res, i_episode+1)) # append res from env to log_res
                

                if (i_episode+1) % args.print_freq == 0:
                    print('Env. {:d}: Return: {:.5f}, ACC: {:.5f}, ACC (full): {:.5f}, Actions: {}, s/episode: {:.3f}'.format(k+1, 
                            ret, acc, acc_full, res['act'], time.time()-t0))
                    # Save logs for each env 
                    save_pickle(log_res[env.seed], path=args.log_dir+'/eval/logs_res_seed%d.p' %(env.seed))
                    avg_acc, gem_bwt = print_log_acc_bwt(res['acc'], res['loss'], 
                                                        output_path=args.log_dir + '/eval', 
                                                        file_name='logs_seed%d.p' %(env.seed), verbose=0) #don't print anything

            if (i_episode+1) % args.print_freq == 0:
                print()
        # Plot figures and log histograms of network params+gradients 
        if agent.learning_started and ((i_episode+1) % args.plot_freq == 0): #and ((i_episode+1) >= args.log_burn_in):
            # Plot agents gradient stats over whoel course of training in all envs
            agent.plot_aggregated_statistics(category='q_value')
            agent.plot_aggregated_statistics(category='q_value_target')
            agent.plot_aggregated_statistics(category='loss')
            agent.plot_aggregated_statistics(category='td_error')
            agent.plot_gradient_statistics()
            agent.logger.plot_scalar('hyper_params', 'epsilon')
            if agent.prioritized_er:
                agent.logger.plot_scalar('hyper_params', 'beta')
            if agent.lr_scheduler:
                agent.logger.plot_scalar('hyper_params', 'lr')
            # Plot ACC with running means
            for k, env in enumerate(valid_envs.envs):
                agent.plot_returns(key='seed%d' %(env.seed), value='ACC')
                #agent.plot_returns(key='seed%d' %(env.seed), value='return')

        # Save logger and checkpoint files
        if ((i_episode+1) % args.checkpoint_freq == 0):
            agent.logger.save_stats('dqn_stats.p')
            # Save checkpoint for agent
            if args.dqn.save_checkpoint:
                state = {'q_net_state_dict': agent.q_net.state_dict(),
                        #'q_net_target_state_dict': agent.q_net_target.state_dict(),
                        'optimizer_state_dict': agent.optimizer.state_dict(),
                        'replay_buffer': agent.replay_buffer,
                        'args': args,
                        }
                if not os.path.exists(args.dqn.checkpoint_dir):
                    os.makedirs(args.dqn.checkpoint_dir, exist_ok=True)
                print('save checkpoint at: ', args.dqn.checkpoint_dir)
                agent.save_checkpoint(state, False, 
                                    checkpoint_dir=args.dqn.checkpoint_dir, 
                                    filename='dqn_checkpoint.pth.tar')

    print('Training DQN done.')
    print()

def make_table_envs(table_dir, obs_dim, action_space, n_envs, seed_start, args):
    envs = []
    count = 0
    for filename in sorted(os.listdir(table_dir)):
        if filename.endswith('.pkl'):
            env = EnvTableFromFile(os.path.join(table_dir, filename), obs_dim, action_space, 
                                                args.dataset, seed_start+count, args, args.shuffle_labels)
            envs.append(env)
            count += 1
            if count == n_envs:
                break
    multi_env = MultiEnvNew(envs)
    return multi_env

def main(args):

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    # Check cuda
    is_cuda = (torch.cuda.is_available() and not args.no_cuda)
    device = torch.device("cuda:0" if is_cuda else "cpu")
    args.device = str(device) # have to cast to string sicne OmegaConf doesn't support pytorch primitives I guess (neither numpy)

    # Data loader
    print('Instantiate data generators and model...')
    # Create envs
    obs_dim = get_observation_dim(args)
    print('Observation dim: ', obs_dim)
    action_space = DiscreteActionSpace(n_tasks=args.n_tasks)
    
    train_envs = make_table_envs(table_dir=args.table_dir + '/train', 
                                obs_dim=obs_dim, 
                                action_space=action_space, 
                                n_envs=args.n_train_envs, 
                                seed_start=args.seed_start_train, 
                                args=args)
    valid_envs = make_table_envs(table_dir=args.table_dir + '/val', 
                                obs_dim=obs_dim, 
                                action_space=action_space, 
                                n_envs=args.n_valid_envs, 
                                seed_start=args.seed_start_valid, 
                                args=args)

    k = 0
    for e in train_envs.envs:
        #print(k, e.dataset.task_ids)
        print(k, e.task_ids)
        k += 1
    for e in valid_envs.envs:
        #print(k, e.dataset.task_ids)
        print(k, e.task_ids)
        k += 1

    accuracies, forgetting, rewards = [], [], []
    n_runs = args.n_runs 
    seed_start = args.seed
    for seed in range(seed_start, seed_start+n_runs):
        print('*' * 100)
        print('\nRun with seed %d: ' %(seed))
        args.dqn.seed = seed
        args.log_dir = args.checkpoint_dir + '/logs/seed%d' %(seed)
        args.dqn.checkpoint_dir = args.checkpoint_dir + '/checkpoint/seed%d' %(seed)

        # Set random seed
        set_random_seed(args.dqn.seed)

        # Create Q-learning agent
        dqn_policy = DQNPolicy(obs_dim, action_space, args)
        agent = DQN(dqn_policy,
                    num_episodes=args.n_episodes, 
                    gamma=args.gamma,
                    lr=args.dqn.lr,
                    opt=args.dqn.optimizer,
                    loss=args.dqn.loss, 
                    buffer_size=args.buffer_size,
                    target_update_freq=args.target_update_freq, 
                    exploration_fraction=args.exploration_fraction,
                    exploration_initial_eps=args.exploration_start_eps,
                    exploration_final_eps=args.exploration_final_eps,
                    max_grad_norm=args.dqn.max_grad_norm,
                    double_dqn=args.double_dqn,
                    prioritized_er=args.prioritized_er,
                    log_dir=args.log_dir)

        # Run Q-learning
        train(args, agent, train_envs, valid_envs)

    # Print time
    t_elapsed = time.time() - t_start
    print('Elapsed time: {:1f} mins = {:2f} hours'.format(t_elapsed / 60, t_elapsed / 3600))

if __name__ == '__main__':
    print('=' * 100)
    print('Arguments =')
    for arg in vars(args):
        print('\t' + arg + ':', getattr(args, arg))
    print('=' * 100)
    main(args)
    print('All done!')