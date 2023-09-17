
import numpy as np 
import torch

from envs.action_spaces import DiscreteActionSpace
from envs.utils import *
from envs.env_fixed_seed import EnvFixedSeed
from envs.wrappers import EnvWrapperPyTorch

from training.utils import print_log_acc_bwt
from training.vis_utils import plot_task_accuracy, plot_task_proportions

def test_agent_in_env(args, agent, env, split='valid'):
    """ Test agent on new environment.
    """
    # Create shorthands
    n_tasks = env.n_tasks
    device = env.device

    # make some empty lists for logging.
    ep_obs = []          # for observations
    ep_acts = []         # for actions
    ep_rews = []         # list for rewards accrued throughout ep
    ep_rs = []
    
    acc = np.zeros([n_tasks, n_tasks])
    loss = np.zeros([n_tasks, n_tasks])

    # train on first task
    t = 0
    obs_t, info = env.reset() # first obs comes from starting distribution
    ep_obs.append(obs_t)
    
    # Evaluate classifier on test set after first task
    res_t = info[split]
    acc[t, :], loss[t, :] = res_t['acc'][:n_tasks], res_t['loss'][:n_tasks]

    # Step through next task until terminal task T
    for t in range(1, n_tasks):
        # Get action
        #print('obs_t: ', obs_t)
        with torch.no_grad():
            if args.algo == 'ppo' or args.algo == 'a2c':
                _, act_t, _ = agent.predict(obs_t.unsqueeze(0), # already a pytorch tensor
                                            torch.as_tensor(t-1, dtype=torch.long, device=device).unsqueeze(0),
                                            deterministic=True)
            elif args.algo == 'ddpg':
                act_t = agent.predict(torch.as_tensor(obs_t, dtype=torch.float32, device=device).unsqueeze(0),
                                    torch.as_tensor(t-1, dtype=torch.long, device=device).unsqueeze(0),
                                    deterministic=True)
                act_t = agent.action_space.get_index_from_action(t-1, act_t)
            else:
                act_t = agent.predict(torch.as_tensor(obs_t, dtype=torch.float32, device=device).unsqueeze(0),
                                    torch.as_tensor(t-1, dtype=torch.long, device=device).unsqueeze(0),
                                    deterministic=True)
                act_t = act_t.item()
        obs_tp1, rew_t, _, info = env.step(t, act_t)
        
        # save action, reward
        ep_acts.append(int(act_t.item())) if args.algo in ['a2c', 'ppo'] else ep_acts.append(act_t)
        ep_rews.append(rew_t)
        ep_obs.append(obs_tp1)
        ep_rs.append(info['task_proportion'])
        obs_t = obs_tp1

        # Evaluate classifier on test set
        res_t = info[split]
        acc[t, :], loss[t, :] = res_t['acc'][:n_tasks], res_t['loss'][:n_tasks]

    # Prepare outputting results
    res = {}
    res['acc'] = acc
    res['loss'] = loss
    res['return'] = np.sum(ep_rews) # reward comes from validation set
    res['act'] = ep_acts
    res['rs'] = ep_rs
    return res

def test_agent(agent, args):
    # 
    actions = []
    accs = []
    seeds = []
    if not os.path.exists(args.log_dir + '/test'):
        os.makedirs(args.log_dir + '/test', exist_ok=True)

    obs_dim = get_observation_dim(args)
    action_space = DiscreteActionSpace(n_tasks=args.n_tasks)

    seed_start = args.seed_start_test

    for seed_env in range(seed_start, seed_start+args.n_test_envs):
        print('Test seed %d' %(seed_env))
        #set_random_seed(n)
        #dataset = CLDataset(args, args.dataset, seed_env) 
        test_env = EnvFixedSeed(obs_dim, action_space, args.dataset, seed_env, args, shuffle_labels=args.shuffle_labels)
        if args.algo == 'ppo' or args.algo == 'a2c':
            test_env = EnvWrapperPyTorch(test_env, args.device)

        #if args.shuffle_labels:
        #    print(n, test_env.dataset.task_ids)
        res = test_agent_in_env(args, agent, test_env, split='test')
        #res = test(args, agent, test_env, split='test')

        print('Selected Actions: ', res['act'])
        print('Printing schedule used during test:')
        for t, s in enumerate(res['rs'], start=1):
            print(t, s)

        #if args.shuffle_labels:
        #    task_ids = '-'.join([str(tt) for tt in test_env.dataset.task_ids])
        #    fname = 'res_test_seed_{}_taskids_{}.p'.format(n, task_ids)
        #else:
        fname = 'res_test_seed_{}.p'.format(seed_env)
        avg_acc, gem_bwt = print_log_acc_bwt(res['acc'], res['loss'], 
                                            output_path=args.log_dir +'/test', 
                                            file_name=fname)
        print()
        print('*'*100)
        print()

        actions.append(res['act'])
        accs.append(avg_acc)
        seeds.append(seed_env)
        plot_task_accuracy(accs=res['acc'], 
                            save_dir=args.log_dir +'/test', 
                            fname='seed%d_task_acc' %(test_env.seed))
        if args.n_tasks == 5:
            plot_task_proportions(task_proportions=res['rs'], 
                                save_dir=args.log_dir +'/test', 
                                fname='seed%d_task_prop' %(test_env.seed))

    for i in range(len(accs)):
        print('Seed: {}, Actions: {}, ACC: {:.5f}'.format(seeds[i], actions[i], accs[i]))
    print()