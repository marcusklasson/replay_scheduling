
import numpy as np
from itertools import combinations, combinations_with_replacement

class DiscreteActionSpace(object):
    """ Maybe this should be called TaskProportionSpace instead to be more clear. 
    """

    def __init__(self, n_tasks):
        self.n_tasks = n_tasks
        #if n_tasks <= 5:
        #    self.dims = [self.get_dim_at_task(t) for t in range(n_tasks)]
        #self.action_space = dict()
    
    def get_dim_at_task(self, task):
        if task >= 7:
            raise ValueError('The number of actions at task {} are too many for creating the range!'.format(task))
        if task == 0:
            return 0
        x = list(range(task))
        #print('x: ', x)
        #x = list(range(task))
        bins = list(combinations_with_replacement(x, task))
        #print('b: ', bins)
        actions = []
        for b in bins:
            actions.append(b)
            #print(len(b))
        #print()
        #print('actions: ', actions)
        #print('len(actions): ', len(actions))
        return len(actions)

    def _create_actions_given_task(self, task):
        """ Here should the creation of memory compositions exist. 
        """
        if task >= 7:
            raise ValueError('The number of actions at task {} are too many for creating the range!'.format(task))
        #if task == 1: # return action filled with zeros on first task
        #    return [[0.0]*(self.n_tasks-1)]
        t = task
        x = list(range(t))
        actions = list(combinations_with_replacement(x, t)) # create action bins
        #print('create actions: ', actions)
        #actions = []
        for i, bins in enumerate(actions):
            props = [0.0]*t
            values, counts = np.unique(bins, return_counts=True)
            for v, c in zip(values, counts):
                props[v-1] = c / t
            assert np.isclose(sum(props), 1), 'proprtion values = {} should sum to {}'.format(props, 1)
            props = zeropad_list(props, size=self.n_tasks-1)
            actions[i] = props # overwrite bin element with task proportion
        return actions

    def get_actions_at_task(self, task):
        actions = self._create_actions_given_task(task)
        return actions

    def get_action_by_index(self, task, action_index):
        """ This should return something like the memory composition with an index at the task
        """
        actions = self._create_actions_given_task(task)
        d = len(actions)
        if (action_index < 0) or (action_index >= d):
            raise ValueError('Action index {} is not within range 0 <= a < {}'.format(action_index, d))  
        return actions[action_index]

    def get_action_with_equal_proportions(self, task):
        """ task = 1, ..., T
        """
        actions = zeropad_list([], size=self.n_tasks-1)
        for t in range(task):
            actions[t] = 1.0 / (task)
        return actions

    def get_action_with_single_task(self, current_task, wanted_task):
        if current_task <= wanted_task:
            raise ValueError('Current task {} cannot be smaller or equal to the wanted task {}!'.format(current_task, wanted_task))
        action = zeropad_list([], size=self.n_tasks-1)
        action[wanted_task] = 1.0
        return action

    def generate_random_action(self, task):
        """ Generate random action from uniform distribution. 
        """
        n = task # number of memory slots
        random_task_proportion = list(np.random.multinomial(n, [1/n]*n)/n)
        props = zeropad_list(random_task_proportion, size=self.n_tasks-1)
        s = sum(props)
        assert np.isclose(s, 1), 'proprtion values = {} should sum to {}'.format(s, 1)
        return props

class TaskLimitedActionSpace(DiscreteActionSpace):

    def __init__(self, n_tasks, task_sample_limit):
        super().__init__(n_tasks)
        self.task_sample_limit = task_sample_limit

    def get_dim_at_task(self, task):
        if task >= 20:
            raise ValueError('The number of actions at task {} are too many for creating the range!'.format(task))
        if task == 0:
            return 0
        x = list(range(task))
        actions = []
        for m in range(1, self.task_sample_limit+1):
            a = list(combinations(x, m)) # create action bins
            #print('a: ', a)
            actions.extend(a)
        #x = list(range(1, self.memory_limit+1))
        #bins = list(combinations_with_replacement(x, task+1))
        return len(actions)

    def _create_actions_given_task(self, task):
        """ Here should the creation of memory compositions exist. 
        """
        if task >= 10:
            raise ValueError('The number of actions at task {} are too many for creating the range!'.format(task))
        #if task == 1: # return action filled with zeros on first task
        #    return [[0.0]*(self.n_tasks-1)]
        t = task 
        x = list(range(t))
        actions = []
        for m in range(1, self.task_sample_limit+1):
            a = list(combinations(x, m)) # create action bins
            actions.extend(a)
        for i, bins in enumerate(actions):
            props = [0.0]*t
            values, counts = np.unique(bins, return_counts=True)
            for v, c in zip(values, counts):
                props[v] = c / len(bins)
            assert np.isclose(sum(props), 1), 'proprtion values = {} should sum to {}'.format(props, 1)
            #print('in action space: ', props)
            props = zeropad_list(props, size=self.n_tasks-1)
            #print('in action space after: ', props)
            actions[i] = props # overwrite bin element with task proportion
        return actions

    def generate_random_action(self, task):
        """ Generate random action from uniform distribution. 
        """
        t = task # number of memory slots
        m = self.task_sample_limit
        random_task_proportion = list(np.random.multinomial(m, [1/t]*t)/m)
        #print('random_task_prop: ', random_task_proportion)
        props = zeropad_list(random_task_proportion, size=self.n_tasks-1)
        s = sum(props)
        assert np.isclose(s, 1), 'proprtion values = {} should sum to {}'.format(s, 1)
        return props

class TaskActionSpace(DiscreteActionSpace):

    def __init__(self, n_tasks, max_tasks):
        super().__init__(n_tasks)
        self.max_tasks = max_tasks

    def get_dim_at_task(self, task):
        if task <= self.max_tasks:
            return 1
        x = list(range(task))
        bins = list(combinations(x, self.max_tasks)) # create action bins
        n_actions = len(bins)
        return n_actions

    def _create_actions_given_task(self, task):
        """ Here should the creation of memory compositions exist. 
        """
        actions = []
        if task <= self.max_tasks:
            act = list(np.ones(task, dtype=np.float32))
            act = zeropad_list(act, size=self.n_tasks-1)
            assert sum(act) <= self.max_tasks
            actions.append(act)
        else:
            x = list(range(task))
            bins = list(combinations(x, self.max_tasks)) # create action bins
            actions = []
            for b in bins:
                act = np.zeros(task)
                act[list(b)] = 1.0
                act = zeropad_list(list(act), size=self.n_tasks-1)
                assert sum(act) <= self.max_tasks
                actions.append(act)
        return actions
       
    def generate_random_action(self, task):
        """ Generate random action from uniform distribution. 
        """
        n_samples = task if task <= self.max_tasks else self.max_tasks
        sampled_tasks = list(np.random.choice(task, n_samples, replace=False))
        props = np.zeros(task)
        props[sampled_tasks] = 1.0
        #random_task_proportion = list(np.random.multinomial(m, [1.0]*t)/m)
        #print('random_task_prop: ', random_task_proportion)
        props = zeropad_list(list(props), size=self.n_tasks-1)
        assert sum(props) <= self.max_tasks, 'proprtion values = {} should be <= {}'.format(props, self.max_tasks)
        return props
        

def zeropad_list(l, size, padding=0.0):
    return l + [padding] * abs((len(l)-size))

def test():

    n_tasks = 20
    #task_sample_limit = 1
    max_tasks = 7
    action_space = TaskActionSpace(n_tasks, max_tasks) 

    for t in range(1, n_tasks):
        print(t, action_space.get_dim_at_task(t))   
        print()  

    """
    for _ in range(10):
        print('new samples')
        for t in range(1, n_tasks):
            print(t, action_space.generate_random_action(t))   
            print()  
    
    action_space = TaskLimitedActionSpace(n_tasks, task_sample_limit)  
    
    for t in range(1, n_tasks):
        print(t, action_space._create_actions_given_task(t))  

    for t in range(1, n_tasks):
        a = action_space.get_action_with_equal_proportions(t)
        print(a)

    if isinstance(action_space, TaskLimitedActionSpace):
        x = 1
        for t in range(1, n_tasks):
            actions = action_space._create_actions_given_task(t)
            print(t, actions)  
            x *= len(actions)
        print('number of action combinations: ', x)

    for t in range(1, n_tasks):
        a = action_space.get_action_with_single_task(t, 0)
        print(a)
    """

#test()

    