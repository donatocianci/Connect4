import numpy as np
from RL_utils import argmax, coarse_code_board
import torch


#linear TD(lambda) Agent
class NeuralNetTDAgent:
    """
    Initialization of TD(lambda) Agent. All values are set to None so they can
    be initialized in the agent_init method.
    """
    def __init__(self):
        self.lambd = None
        self.gamma = None
        self.delta = None
        self.alpha = None
        self.model = None
        self.prev_features = None
        self.weights = None
        self.columns = None
        self.rows = None
        self.inarow = None


    def agent_init(self, agent_info={}):
        """Setup for the agent called when the experiment first starts."""
        self.lambd = agent_info.get("lambda", 0.7)
        self.gamma = agent_info.get("gamma", 0.5)
        self.alpha = agent_info.get("alpha", 0.01)
        self.columns = agent_info.get("columns", 7)
        self.rows = agent_info.get("rows", 6)
        self.inarow = agent_info.get("inarow", 4)
        num_features = int( 2 * (self.columns + self.rows + 2 * (self.rows + self.columns - 1)) )
        self.weights = agent_info.get("weights", None )
        self.delta = 0.1
        self.model = torch.nn.Sequential(
                        torch.nn.Linear(num_features, num_features + 5),
                        torch.nn.ReLU(),
                        torch.nn.Linear(num_features + 5, num_features + 5),
                        torch.nn.ReLU(),
                        torch.nn.Linear(num_features + 5, 1),
                        torch.nn.Sigmoid()
                        )

    def agent_start_episode(self):

        num_features = int( 2 * (self.columns + self.rows + 2 * (self.rows + self.columns - 1)) )
        x_init = np.zeros(num_features)
        self.model( torch.from_numpy(x_init).unsqueeze(0).float() ).backward()
        self.model.zero_grad()


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

        with torch.no_grad():

            for c in possible_actions:
                possible_board = test_drop(board, c)
                X = coarse_code_board(possible_board, mark, rows, columns)
                possible_features.append( X )
                action_values.append( self.model( torch.from_numpy(X).unsqueeze(0).float() ).item() )

        action = possible_actions[argmax(action_values)]
        features = possible_features[argmax(action_values)]

        self.prev_features = torch.from_numpy( features )

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


        for p in self.model.parameters():
            p.grad *= self.lambd * self.gamma

        v_prev = self.model( self.prev_features.unsqueeze(0).float() )
        v_prev.backward()


        with torch.no_grad():
            #get the new features
            current_features = coarse_code_board(board, mark, rows, columns)
            current_v = self.model( torch.from_numpy(current_features).unsqueeze(0).float() ).item()

            self.delta = reward + self.gamma * current_v - v_prev.item()

            for param in self.model.parameters():
                param += self.alpha * self.delta * param.grad

    def last_agent_update(self, reward):

        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            state (Numpy array): the state observation from the
                environment's evn_start function.
        Returns:
            The first action the agent takes.
        """

        for p in self.model.parameters():
            p.grad *= self.lambd * self.gamma

        v_prev = self.model( self.prev_features.unsqueeze(0).float() )
        v_prev.backward()

        with torch.no_grad():

            self.delta = reward - v_prev.item()

            for param in self.model.parameters():
                param += self.alpha * self.delta * param.grad

class EnsembleTDAgent():

    def __init__(self):
        self.tau = None
        self.agents = None

    def agent_init(self, agent_info = {}):
        self.tau 
