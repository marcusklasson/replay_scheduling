
class State(object):

    def __init__(self, n_tasks, task, action_space, actions=[], replay_schedule=[]):
        if task > n_tasks:
            raise ValueError('task cannot be greater than number of tasks!')
        self.n_tasks = n_tasks
        self.task = task
        self.action_space = action_space
        #self.task_proportion = task_proportion
        self.replay_schedule = replay_schedule
        self.actions = actions
        #print(self.replay_schedule)
        #print(replay_schedule)

    def move(self, action, action_index=None):
        next_rs = self.replay_schedule.copy()
        #if self.task > 1:
        next_rs.append(action)
        #print('in move, next rs: ', next_rs)
        actions = self.actions.copy()
        if action_index is not None:
            actions.append(action_index)
        return State(self.n_tasks, 
                    self.task+1, 
                    self.action_space, 
                    actions=actions, 
                    replay_schedule=next_rs)

    def get_legal_actions(self):
        actions = self.action_space.get_actions_at_task(self.task)
        #print('legal actions: ', actions)
        return actions

    def get_random_action(self):
        random_action = self.action_space.generate_random_action(self.task)
        return random_action

    def is_terminal_state(self):
        #print('terminal state!!: task {}/{} - {}'.format(self.task, self.n_tasks, self.task < self.n_tasks))
        if self.task < self.n_tasks: #self.n_tasks-1:
            return False
        return True 

    def get_replay_schedule(self):
        return self.replay_schedule.copy(), self.actions.copy()

    def result(self):
        """ Perhaps this could pass the actions, but I could probably name it better than 'game_result'
        """
        pass