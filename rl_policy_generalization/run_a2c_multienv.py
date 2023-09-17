
import os, argparse, time
import numpy as np
from omegaconf import OmegaConf

import torch

from training.utils import print_log_acc_bwt, set_random_seed, save_pickle
from training.vis_utils import plot_task_accuracy, plot_replay_schedule_bubble_plot
from envs.wrappers import MultiEnv, MultiEnvWrapperPyTorch, EnvWrapperPyTorch, MultiEnvNew
from envs.action_spaces import DiscreteActionSpace
from envs.utils import *
from envs.env_fixed_seed import EnvFixedSeed
from envs.env_table import EnvTable, EnvTableFromFile

from rl.a2c.a2c import A2C 
from rl.a2c.model import ActorCriticPolicy 
from rl.utils import test_agent, test_agent_in_env

t_start = time.time()

# Arguments
parser = argparse.ArgumentParser(description='Coding...')
# Load the config file
parser.add_argument('--config',  type=str, default='./configs/a2c/new_task_orders/mnist.yml')
flags =  parser.parse_args()
args = OmegaConf.load(flags.config)

print()

########################################################################################################################

def train(args, model, train_envs, valid_envs):

    obs, _ = train_envs.reset() # reset only needed once since obs is the same at all time steps in each env
    for proc, ob in enumerate(obs):
        model.rollouts.obs[0, proc].copy_(ob)
    model.rollouts.to(model.device)

    start = time.time()
    num_episodes = args.n_episodes
    num_steps = args.a2c.n_steps
    num_processes = len(train_envs) #args.num_processes

    val_seeds = [env.seed for env in valid_envs.envs]
    log_res = {seed: [] for seed in val_seeds}
    
    accs = {'train': [[]]*len(train_envs), 
            'valid': [[]]*len(valid_envs),
            }

    print('Start training...')
    print()
    for j in range(num_episodes):

        for step in range(num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob = model.predict(model.rollouts.obs[step], 
                                                                model.rollouts.task_ids[step]-1,
                                                                deterministic=False)
            # Obser reward and next obs
            t = model.rollouts.task_ids[step]#.item()
            obs, reward, done, info = train_envs.step(t, action)
            if done[0]:
                obs, _ = train_envs.reset()
            # If done then clean the history of observations.
            masks = torch.ones([num_processes, 1]).float()*(0.0 if done[0] else 1.0)
            task_ids = torch.ones([num_processes, 1]).long()*(1 if done[0] else t[0].item()+1)
            model.insert_rollout(obs, action, action_log_prob, value, reward, masks, task_ids=task_ids)
        
        # Update model after number of steps
        batch_summary = model.update_parameters()
        if (model.n_updates % args.log_freq == 0):
            model.log_scalar('train', 'entropy', batch_summary['entropy'], it=model.n_updates)
            model.log_scalar('train', 'value', batch_summary['value'], it=model.n_updates)
            model.log_scalar('train', 'value_loss', batch_summary['value_loss'], it=model.n_updates)
            model.log_scalar('train', 'policy_loss', batch_summary['policy_loss'], it=model.n_updates)
            model.log_scalar('train', 'grad_norm', batch_summary['grad_norm'], it=model.n_updates)
            if args.algo == 'ppo':
                model.log_scalar('train', 'approx_kl', batch_summary['approx_kl'], it=model.n_updates)
        
        if args.verbose > 1:
            #print('rollouts.obs: ', rollouts.obs.squeeze(1))
            print('rollouts.masks: ', model.rollouts.masks.view(-1))
            print('rollouts.returns: ', model.rollouts.returns.view(-1))
            print('rollouts.rewards: ', model.rollouts.rewards.view(-1))
            print('rollouts.task_ids: ', model.rollouts.task_ids.view(-1))
            print('rollouts.actions: ', model.rollouts.actions.view(-1))

        if (j+1) % args.print_freq == 0:
            total_num_steps = (j + 1) * num_processes * args.n_tasks
            end = time.time()
            print('Episodes {} '.format(j+1))
            print('Entropy: {:.3f} \nMean Value: {:.3f}\nValue loss: {:.3f} \nPolicy loss: {:.3f}'.format(
                    batch_summary['entropy'], batch_summary['value'], batch_summary['value_loss'], batch_summary['policy_loss']))
            

        if (args.eval_freq is not None and (j+1) % args.eval_freq == 0):
            if not os.path.exists(args.log_dir + '/eval'):
                os.makedirs(args.log_dir + '/eval', exist_ok=True)
            if (j+1) % args.print_freq == 0:
                print('Validation envs:')
            for k, env in enumerate(valid_envs.envs):
                t0 = time.time()
                res = test_agent_in_env(args, model, EnvWrapperPyTorch(env, env.device), split='test') 
                ret, acc = res['return'], np.mean(res['acc'][-1])
                model.log_scalar('valid_env%d' %(k+1), 'return', ret, j+1)
                model.log_scalar('valid_env%d' %(k+1), 'ACC', acc, j+1)
                for t, action in enumerate(res['act'][1:], start=2):
                    model.log_scalar('seed%d' %(env.seed), 'actions/task%d'%(t+1), action, j+1)
                plot_task_accuracy(accs=res['acc'], 
                                    save_dir=args.log_dir + '/eval', 
                                    fname='valid_task_acc_seed%d' %(env.seed))
                if args.n_tasks == 5:
                    plot_replay_schedule_bubble_plot(replay_schedule=res['rs'],  
                                        save_dir=args.log_dir + '/eval', 
                                        fname='valid_task_prop_seed%d' %(env.seed))
                log_res[env.seed].append((res, j+1)) # append res from env to log_res
                if (j+1) % args.print_freq == 0:
                    print('Env. {:d}, Seed {:d} - Return: {:.5f}, ACC: {:.5f}, Actions: {}, s/episode: {:.3f}'.format(k+1, 
                            env.seed, ret, acc, res['act'], time.time()-t0))
                    # Save logs for each env 
                    save_pickle(log_res[env.seed], path=args.log_dir+'/eval/logs_res_seed%d.p' %(env.seed))
                    avg_acc, gem_bwt = print_log_acc_bwt(res['acc'], res['loss'], 
                                                        output_path=args.log_dir + '/eval', 
                                                        file_name='logs_seed%d.p' %(env.seed), verbose=0) #don't print anything
            if (j+1) % args.print_freq == 0: # add extra print 
                print()

        if (args.plot_freq is not None and (j+1) % args.plot_freq == 0):
            model.logger.plot_scalar('train', 'value_loss')
            model.logger.plot_scalar('train', 'policy_loss')
            model.logger.plot_scalar('train', 'entropy')
            model.logger.plot_scalar('train', 'value')
            model.logger.plot_scalar('train', 'grad_norm')

            for k in range(len(valid_envs)):
                model.plot_returns(key='valid_env%d' %(k+1), value='ACC')
                #model.plot_returns(key='valid_env%d' %(k+1), value='return')

        # save for every interval-th episode or for the last epoch
        if ((j+1) % args.save_interval == 0
                or (j+1) == num_episodes) and args.a2c.checkpoint_dir != "":
            save_path = args.a2c.checkpoint_dir
            try:
                os.makedirs(save_path, exist_ok=True)
            except OSError:
                pass
            torch.save(model.actor_critic, os.path.join(save_path, "actor_critic.pth.tar"))
            model.logger.save_stats('%s_stats.p' %(args.algo))

    print('Training done.')
    print()
    
def make_table_envs(table_dir, obs_dim, action_space, n_envs, seed_start, args):
    envs = []
    count = 0
    for filename in os.listdir(table_dir):
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
    os.makedirs(args.log_dir, exist_ok=True)
    # Check cuda
    is_cuda = (torch.cuda.is_available() and not args.no_cuda)
    device = torch.device("cuda:0" if is_cuda else "cpu")
    args.device = str(device) # have to cast to string sicne OmegaConf doesn't support pytorch primitives I guess (neither numpy)

    # Data loader
    print('Instantiate data generators and model...')
    # Set random seed
    set_random_seed(args.seed)
    # Create envs
    obs_dim = get_observation_dim(args)
    action_space = DiscreteActionSpace(n_tasks=args.n_tasks)
    n_actions = action_space.max_dim

    train_envs = make_table_envs(table_dir=args.table_dir + '/train', 
                                obs_dim=obs_dim, 
                                action_space=action_space, 
                                n_envs=args.n_train_envs, 
                                seed_start=args.seed_start_train, 
                                args=args)
    train_envs = MultiEnvWrapperPyTorch(train_envs, args.device)
    valid_envs = make_table_envs(table_dir=args.table_dir + '/val', 
                                obs_dim=obs_dim, 
                                action_space=action_space, 
                                n_envs=args.n_valid_envs, 
                                seed_start=args.seed_start_valid, 
                                args=args)
    valid_envs = MultiEnvWrapperPyTorch(valid_envs, args.device)

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
        args.a2c.seed = seed
        args.log_dir = args.checkpoint_dir + '/logs/a2c_seed%d' %(seed)
        os.makedirs(args.log_dir, exist_ok=True)
        args.a2c.checkpoint_dir = args.checkpoint_dir + '/checkpoint/a2c_seed%d' %(seed)
        os.makedirs(args.a2c.checkpoint_dir, exist_ok=True)

        # Set random seed
        set_random_seed(args.a2c.seed)

        # Create policy
        ac_policy = ActorCriticPolicy(obs_dim, action_space, args)
        n_steps = args.a2c.n_steps
        model = A2C(ac_policy, 
                    n_processes=args.n_train_envs, 
                    n_steps=args.a2c.n_steps, 
                    value_loss_coef=args.value_loss_coef, 
                    entropy_coef=args.entropy_coef, 
                    discount=args.discount, 
                    lr=args.a2c.lr, 
                    max_grad_norm=args.max_grad_norm, 
                    use_gae=args.use_gae, 
                    gae_lambda=args.gae_lambda, 
                    use_proper_time_limits=args.use_proper_time_limits, 
                    device=args.device, 
                    log_dir=args.log_dir)

        # Run A2C 
        train(args, model, train_envs, valid_envs)

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