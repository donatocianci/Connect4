import numpy as np
from RL_utils import argmax, coarse_code_board

#linear TD(lambda) Agent
class TDAgent:
    """
    Implementation of a linear TD(lambda) Agent. 
    """
    def __init__(self, agent_info = {}):
        """Setup for the agent called when the experiment first starts."""
        self.lambd = agent_info.get("lambda", 0.7)
        self.gamma = agent_info.get("gamma", 0.5)
        self.alpha = agent_info.get("alpha", 0.01)
        self.epsilon = agent_info.get("epsilon", 0.0)
        self.columns = agent_info.get("columns", 4)
        self.rows = agent_info.get("rows", 4)
        self.inarow = agent_info.get("inarow", 4)
        num_features = 150
        self.w = agent_info.get("w", np.random.normal( 50.0,25.0, size = num_features ) )
        self.z = np.zeros(num_features)
        self.delta = 0.1

    def agent_start_episode(self):
        num_features = 150
        self.z = np.zeros(num_features)

    def select_action(self, state):
        """
        Selects an action using epsilon greedy
        Args:
        state - dictionary that contains the board state and current turn's mark.
        Returns:
        int between 0 and 6 denoting the column to play.
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

        return int(action)

    def agent_update(self, reward, state):
        """Updates the weights of the model based off of the observed reward and
        the current state using the TD(lambda) update rule.
        Args:
            reward -- Float of the observed reward for the test_board
            state -- dictionary containing board configuration and current turn's
            mark, accessed through keys: 'board' and 'mark'
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

        """If the episode is terminal, then we change the update rule
        """

        #update agent weights
        self.z = self.gamma * self.lambd * self.z + self.prev_features
        self.delta = reward - np.dot(self.w, self.prev_features)
        self.w = self.w + self.alpha * self.delta * self.z
