
import numpy as np
import itertools 
from itertools import combinations, combinations_with_replacement

def zeropad_list(l, size, padding=0.0):
    return l + [padding] * abs((len(l)-size))

class DiscreteActionSpace(object):

    def __init__(self, n_tasks, seed=None):
        self.n_tasks = n_tasks
        self.seed = seed
        if seed is None:
            rs = None #np.random.RandomState()
        else:
            rs = np.random.RandomState(seed)
        self.rs = rs
        self._make_dimensions()
        #self._make_actions()
    
    @property
    def max_dim(self):
        return self.dims[-1]

    @property
    def shape(self):
        return 1 

    def get_dim_at_task(self, task):
        if task >= self.n_tasks:
            raise ValueError('Task {} cannot begreater or equal to n_tasks {}!'.format(task, self.n_tasks))
        elif task == self.n_tasks-1:
            return self.dims[-1]
        return self.dims[task]

    def _make_dimensions(self):
        n_tasks = self.n_tasks
        if self.n_tasks >= 7:
            n_tasks = 7
            print('The number of actions are too many to create all dimensions if n_tasks={} is greater than 7!'.format(self.n_tasks))
            #raise ValueError('The number of actions are too many if n_tasks={} is greater than 7!'.format(self.n_tasks))
        self.dims = []
        for t in range(1, n_tasks):
            x = list(range(t))
            bins = list(combinations_with_replacement(x, t))
            #if (t > 1) and (t % 2!=0):
            #    bins1 = list(combinations(x, t-1)) 
            #    bins.extend(bins1)
            self.dims.append(len(bins))
    
    """
    def _make_actions(self):
        n_tasks = self.n_tasks
        if self.n_tasks >= 7:
            n_tasks = 7
            print('The number of actions are too many to create all dimensions if n_tasks={} is greater than 7!'.format(self.n_tasks))
        self.actions, self.dims = [], []
        for t in range(1, n_tasks):
            x = list(range(t))
            acts_t = list(combinations_with_replacement(x, t)) # create action bins
            #print('acts_t: ', acts_t)
            
            if (t > 1) and (t % 2!=0):
                acts_t1 = list(combinations(x, t-1)) 
                acts_t.extend(acts_t1)
            

            for i, bins in enumerate(acts_t):
                props = [0.0]*t
                values, counts = np.unique(bins, return_counts=True)
                for v, c in zip(values, counts):
                    props[v] = c / len(bins) #t 
                assert np.isclose(sum(props), 1), 'proprtion values = {} should sum to {}'.format(props, 1)
                props = zeropad_list(props, size=self.n_tasks-1)
                acts_t[i] = props # overwrite bin element with task proportion
            
                if props not in self.actions:
                    self.actions.append(props)
            print(self.actions)
            d = len(self.actions)
            self.dims.append(d)
    """    
    

    def sample(self, task):
        if self.rs is None:
            action_index = np.random.randint(self.get_dim_at_task(task))
        else:
            action_index = self.rs.randint(self.get_dim_at_task(task))
        actions = self._create_actions_given_task(task)
        return action_index, actions[action_index]

    
    def _create_actions_given_task(self, task):
        # creates all possible task proportions (actions) at the given task 
        # task = 0, ..., n_tasks-1
        if task >= 7:
            raise ValueError('The number of actions at task {} are too many for creating the range!'.format(task))
        t = task+1
        x = list(range(t))
        actions = list(combinations_with_replacement(x, t)) # create action bins

        # Not sure if I should add these actions, because action space construction becomes very different and space gets larger
        #if (t > 1) and (t % 2!=0):
        #    actions_equal = list(combinations(x, t-1)) 
        #    actions.extend(actions_equal)

        #print(actions)
        for i, bins in enumerate(actions):
            props = [0.0]*t
            values, counts = np.unique(bins, return_counts=True)
            for v, c in zip(values, counts):
                props[v] = c / len(bins) #t 
            assert np.isclose(sum(props), 1), 'proprtion values = {} should sum to {}'.format(props, 1)
            props = zeropad_list(props, size=self.n_tasks-1)
            actions[i] = props # overwrite bin element with task proportion
        #print(actions)
        return actions
    

    def get_actions_at_task(self, task):
        if task >= len(self.dims):
            task = len(self.dims)-1
        actions = self._create_actions_given_task(task)
        #d = self.get_dim_at_task(task)
        #actions = self.actions[:d]
        return actions

    def get_action_by_index(self, task, action_index):
        # create and return single task proportion (action) given task and index
        actions = self._create_actions_given_task(task)
        d = len(actions)
        if (action_index < 0) or (action_index > d):
            raise ValueError('Action index {} is not within range 0 <= a < {}'.format(action_index, d))  
        return actions[action_index]

    def get_action_with_equal_proportions(self, task):
        # get equal task proportion given task, used in ETS baseline code
        # task = 0, ..., n_tasks-1
        actions = zeropad_list([], size=self.n_tasks-1)
        task = task+1
        for t in range(task):
            actions[t] = 1.0 / (task)
        return actions

    def get_action_with_single_task(self, current_task, wanted_task):
        # get task proportion with 1 at wanted task and zero at other actions
        if current_task <= wanted_task:
            raise ValueError('Current task {} cannot be smaller or equal to the wanted task {}!'.format(current_task, wanted_task))
        action = zeropad_list([], size=self.n_tasks-1)
        action[wanted_task] = 1.0
        return action

    def generate_random_action(self, task):
        # generate a random task proportion at given task, commonly used by datasets with long task horizon (10-20 tasks)
        # task = 0, ..., n_tasks-1
        n = task+1 # number of memory slots
        random_task_proportion = list(np.random.multinomial(n, [1/n]*n)/n)
        props = zeropad_list(random_task_proportion, size=self.n_tasks-1)
        s = sum(props)
        assert np.isclose(s, 1), 'proprtion values = {} should sum to {}'.format(s, 1)
        return props   

    def reset_rng(self):
        if self.seed is None:
            self.rs = np.random.RandomState()
        else:
            self.rs = np.random.RandomState(self.seed)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

class ContinuousActionSpace(object):

    def __init__(self, n_tasks, a_low=-1, a_high=1.0, seed=None):
        self.n_tasks = n_tasks
        self.a_low = a_low 
        self.a_high = a_high
        self.seed = seed
        if seed is None:
            rs = None #np.random.RandomState()
        else:
            rs = np.random.RandomState(seed)
        self.rs = rs
    
    @property
    def max_dim(self):
        return self.n_tasks-1

    @property
    def shape(self):
        return self.n_tasks-1

    def _get_proportion(self, action, low=0.0, high=1.0):
        # transform to range [-1, 1] -> [0, 1]
        props = (action + 1) * (high-low)/2.0 + low 
        return props

    def _get_action(self, props, low=-1.0, high=1.0):
        # transform to range [0, 1] -> [-1, 1]
        actions = props * (high-low)/1.0 + low 
        return actions 

    def get_task_proportion_from_action(self, action, task):
        if isinstance(action, list):
            action = np.array(action)
        # set illegal actions to large negative value before softmax
        offset = task + 1
        action[offset:] = -1e10
        #print('action: ', action)
        #props_hat = self._get_proportion(action)
        props = softmax(action)
        #print('props: ', props)
        #print()
        #offset = task + 1
        # mask out illegal proportions
        #props_hat[offset:] = 0.0
        # normalize legal actions
        #props = props_hat / np.sum(props_hat)
        assert np.isclose(np.sum(props), 1), 'proportion values = {} should sum to {}'.format(props, 1)
        return props 

    def get_action_with_equal_proportions(self, task):
        # get equal task proportion given task, used in ETS baseline code
        # task = 0, ..., n_tasks-1
        actions = zeropad_list([], size=self.n_tasks-1)
        task = task+1
        for t in range(task):
            actions[t] = 1.0 / (task)
        return actions

    def get_action_with_single_task(self, current_task, wanted_task):
        # get task proportion with 1 at wanted task and zero at other actions
        if current_task <= wanted_task:
            raise ValueError('Current task {} cannot be smaller or equal to the wanted task {}!'.format(current_task, wanted_task))
        action = zeropad_list([], size=self.n_tasks-1)
        action[wanted_task] = 1.0
        return action

    def generate_random_action(self, task):
        # generate a random task proportion at given task, commonly used by datasets with long task horizon (10-20 tasks)
        # task = 0, ..., n_tasks-1
        n = task+1 # number of memory slots
        random_task_proportion = list(np.random.uniform(low=0.0, high=1.0, size=n))
        props_hat = zeropad_list(random_task_proportion, size=self.n_tasks-1)
        props = props_hat / np.sum(props_hat)
        s = sum(props)
        assert np.isclose(s, 1), 'proprtion values = {} should sum to {}'.format(s, 1)
        return props   

    def reset_rng(self):
        if self.seed is None:
            self.rs = np.random.RandomState()
        else:
            self.rs = np.random.RandomState(self.seed)

class CountingActionSpace(DiscreteActionSpace):
    
    def __init__(self, n_tasks, seed=None):
        super().__init__(n_tasks, seed)

    def _make_dimensions(self):
        n_tasks = self.n_tasks
        """
        if self.n_tasks >= 7:
            n_tasks = 7
            print('The number of actions are too many to create all dimensions if n_tasks={} is greater than 7!'.format(self.n_tasks))
            #raise ValueError('The number of actions are too many if n_tasks={} is greater than 7!'.format(self.n_tasks))
        """
        self.dims = []
        for t in range(1, n_tasks):
            x = [list(i) for i in itertools.product([0, 1], repeat=t)]
            x.pop(0) # remove the action with only zeros
            #print(t, x, len(x))
            self.dims.append(len(x))
        #print(self.dims)

    def _create_actions_given_task(self, task):
        # creates all possible task proportions (actions) at the given task 
        # task = 0, ..., n_tasks-1
        """
        if task >= 7:
            raise ValueError('The number of actions at task {} are too many for creating the range!'.format(task))
        """
        t = task+1
        actions = [list(reversed(i)) for i in itertools.product([0, 1], repeat=t)] # reversing to get same actions on increasing dims
        actions.pop(0)

        for i, counts in enumerate(actions):
            props = [0.0]*t
            for j, c in enumerate(counts):
                props[j] = c / sum(counts) 
            assert np.isclose(sum(props), 1), 'proprtion values = {} should sum to {}'.format(props, 1)
            props = zeropad_list(props, size=self.n_tasks-1)
            actions[i] = props # overwrite bin element with task proportion
        return actions



def test():

    n_tasks = 5
    #task_sample_limit = 1
    max_tasks = 7
    #action_space = DiscreteActionSpace(n_tasks)
    action_space = CountingActionSpace(n_tasks) 

    for t in range(0, n_tasks-1):
        act_dim = action_space.get_dim_at_task(t)
        print(t, act_dim)
        print('random_action: ', action_space.get_action_by_index(t, np.random.randint(act_dim)))   
        print()  

    for t in range(0, n_tasks-1):
        print(t, action_space._create_actions_given_task(t))  
        #print(t, action_space.get_actions_at_task(t))   
        if t > 1:
            action_index = 5
            print(action_index, action_space.get_actions_at_task(t)[action_index]) 
    print()
    print('equal tasks:')
    for t in range(0, n_tasks-1):
        a = action_space.get_action_with_equal_proportions(t)
        print(a)
    print()

#test()    
