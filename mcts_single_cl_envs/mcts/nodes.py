import numpy as np
from collections import defaultdict
from abc import ABC, abstractmethod


class MonteCarloTreeSearchNode(ABC):

    def __init__(self, state, parent=None):
        """
        Parameters
        ----------
        state : mcts.state.State 
        parent : MonteCarloTreeSearchNode
        """
        self.state = state
        self.parent = parent
        self.children = []

    @property
    @abstractmethod
    def untried_actions(self):
        """

        Returns
        -------
        list of mctspy.games.common.AbstractGameAction

        """
        pass

    @property
    @abstractmethod
    def q(self):
        pass

    @property
    @abstractmethod
    def n(self):
        pass

    @abstractmethod
    def expand(self):
        pass

    @abstractmethod
    def is_terminal_node(self):
        pass

    @abstractmethod
    def rollout(self):
        pass

    @abstractmethod
    def backpropagate(self, reward):
        pass

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c_param=1.4):
        choices_weights = [
            (c.q / c.n) + c_param * np.sqrt((2 * np.log(self.n) / c.n))
            for c in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves):        
        return np.random.randint(len(possible_moves)) #possible_moves[np.random.randint(len(possible_moves))]

class ReplaySchedulingNode(MonteCarloTreeSearchNode):

    def __init__(self, state, parent=None):
        super().__init__(state, parent)
        self._number_of_visits = 0.
        self._results = [] 
        self._untried_actions = None

    @property
    def untried_actions(self):
        if self._untried_actions is None:
            self._untried_actions = self.state.get_legal_actions()
        return self._untried_actions

    @property
    def q(self):
        return self._results # returns list with all results

    @property
    def n(self):
        return self._number_of_visits

    def rollout_policy(self, possible_moves):
        index = np.random.randint(len(possible_moves))     
        #print(index, len(possible_moves))   
        return possible_moves[index], index

    def get_untried_action(self):
        untried_actions = self.untried_actions
        action_index = len(untried_actions)-1
        action = untried_actions.pop(-1)
        return action, action_index

    def expand(self):
        #action = self.untried_actions.pop()
        action, action_index = self.get_untried_action()
        #print(action)
        next_state = self.state.move(action, action_index)
        child_node = ReplaySchedulingNode(
            next_state, parent=self
        )
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        return self.state.is_terminal_state()

    def rollout(self):
        current_rollout_state = self.state
        while not current_rollout_state.is_terminal_state():
            possible_moves = current_rollout_state.get_legal_actions()
            action, index = self.rollout_policy(possible_moves)
            current_rollout_state = current_rollout_state.move(action, index)
        #return current_rollout_state.result
        rs, actions = current_rollout_state.get_replay_schedule()
        rollout_res = {'rs': rs, 'actions': actions}
        return rollout_res # return replay schedules for training

    def backpropagate(self, result):
        self._number_of_visits += 1.
        self._results.append(result)
        if self.parent:
            self.parent.backpropagate(result)

    def best_child(self, c_param=0.1):
        """ Get best child node selected based on UCT score from Feature Selection paper (Chaudhry).
            Args:
                c_param (float): Exploration constant 
            Returns:
                (ReplaySchedulingNode): Best child
        """
        choices_weights = [
            np.max(c.q) + c_param * np.sqrt((2 * np.log(self.n) / c.n))
            for c in self.children
        ]
        return self.children[np.argmax(choices_weights)]

class LongTaskHorizonNode(ReplaySchedulingNode):
    """ Use this when number of tasks are larger than 5 if using the our proposed discretization of the action space. 
        The action space during the rollouts might be so large that we have to generate a random action with a uniform distribution.
    """
    
    def __init__(self, state, parent=None):
        super().__init__(state, parent)
    """
    def expand(self):
        action = self.untried_actions.pop()
        next_state = self.state.move(action)
        child_node = LongTaskHorizonNode(
            next_state, parent=self
        )
        self.children.append(child_node)
        return child_node
    """

    def expand(self):
        #action = self.untried_actions.pop()
        action, action_index = self.get_untried_action()
        #print('expanmd! ', action)
        next_state = self.state.move(action, action_index)
        child_node = LongTaskHorizonNode(
            next_state, parent=self
        )
        self.children.append(child_node)
        return child_node

    def rollout(self):
        current_rollout_state = self.state
        #print('rs in state: ', current_rollout_state.replay_schedule)
        #print('in rollout')
        while not current_rollout_state.is_terminal_state():
            #print('inside rollout loop')
            if current_rollout_state.task >= 7:
                action = current_rollout_state.get_random_action()
                index = None 
            else:
                possible_moves = current_rollout_state.get_legal_actions()
                action, index = self.rollout_policy(possible_moves)
            current_rollout_state = current_rollout_state.move(action, index)
            #current_rollout_state = current_rollout_state.move(action)
        #print()
        rs, actions = current_rollout_state.get_replay_schedule()
        #print('after rollout')
        #print('actions: ', actions)
        #print('rs: ', rs)
        #print()
        rollout_res = {'rs': rs, 'actions': actions}
        return rollout_res #current_rollout_state.get_replay_schedule()

