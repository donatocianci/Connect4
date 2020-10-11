import numpy as np
from RL_utils import argmax, coarse_code_board

#linear TD(lambda) Agent
class TDAgent:
    """
    Initialization of TD(lambda) Agent. All values are set to None so they can
    be initialized in the agent_init method.
    """
    def __init__(self):
        self.lambd = None
        self.gamma = None
        self.delta = None
        self.w = None
        self.alpha = None
        self.prev_features = None
        self.columns = None
        self.rows = None
        self.inarow = None
        self.z = None
        self.epsilon = None

    def agent_init(self, agent_info={}):
        """Setup for the agent called when the experiment first starts."""
        self.lambd = agent_info.get("lambda", 0.7)
        self.gamma = agent_info.get("gamma", 0.5)
        self.alpha = agent_info.get("alpha", 0.1)
        self.epsilon = agent_info.get("epsilon", 0.2)
        self.columns = agent_info.get("columns", 7)
        self.rows = agent_info.get("rows", 6)
        self.inarow = agent_info.get("inarow", 4)
        num_features = 2 * (self.columns + self.rows + 2 * (self.rows + self.columns - 1))
        self.w = agent_info.get("w", np.random.normal( 50.0,25.0, size = num_features ) )
        self.z = np.zeros(num_features)
        self.delta = 0.1

    def agent_start_episode(self):
        num_features = 2 * (self.columns + self.rows + 2 * (self.rows + self.columns - 1))
        self.z = np.zeros(num_features)

    def select_action(self, state):
        """
        Selects an action using epsilon greedy
        Args:
        tiles - np.array, an array of active tiles
        Returns:
        (chosen_action, action_value) - (int, float), tuple of the chosen action
                                        and it's value
        """

        board = state['board']
        columns = self.columns
        rows = self.rows
        mark = state['mark']

        possible_actions = [c for c in range(columns) if board[c] == 0]
        action_values = []
        possible_features = []

        def test_drop(board, column):
            test_board = np.copy(board)
            row = max([r for r in range(rows) if board[column + (r * columns)] == 0])
            test_board[column + (row * columns)] = mark

            return test_board

        for c in possible_actions:
            possible_board = test_drop(board, c)
            X = coarse_code_board(possible_board, mark, rows, columns)
            possible_features.append(X)
            action_values.append( np.dot(self.w,X) )

        if np.random.random() < self.epsilon:
            action = np.random.choice(possible_actions)
            features = possible_features[possible_actions.index(action)]

        else:
            action = possible_actions[argmax(action_values)]
            features = possible_features[argmax(action_values)]

        self.prev_features = features

        return action

    def agent_update(self, reward, state):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            state (Numpy array): the state observation from the
                environment's evn_start function.
        Returns:
            The first action the agent takes.
        """
        board = state['board']
        columns = self.columns
        rows = self.rows
        mark = state['mark']

        #get the new features
        current_features = coarse_code_board(board, mark, rows, columns)
        current_v = np.dot(self.w, current_features )

        #update agent weights
        self.z = self.gamma * self.lambd * self.z + self.prev_features
        self.delta = reward + self.gamma * current_v - np.dot(self.w, self.prev_features)
        self.w += self.alpha * self.delta * self.z

    def last_agent_update(self, reward):

        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            state (Numpy array): the state observation from the
                environment's evn_start function.
        Returns:
            The first action the agent takes.
        """

        #update agent weights
        self.z = self.gamma * self.lambd * self.z + self.prev_features
        self.delta = reward - np.dot(self.w, self.prev_features)
        self.w = self.w + self.alpha * self.delta * self.z
