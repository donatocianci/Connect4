import numpy as np
from RL_utils import argmax, coarse_code_board
import torch


from RL_utils import train, watch_play, evaluate, argmax, check_winning_move
from eval_agents import BaseAgent, StepPlay, McGoo
from random import choice
import numpy as np
import torch

class ConnectNet4(torch.nn.Module):
    #set learning rate to 1e-2
    #512 batch
    def __init__(self):

        super(ConnectNet4, self).__init__()

        self.features = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels = 2, out_channels = 128, kernel_size = 3, padding = 1),
                        torch.nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 2, stride = 2),
                        torch.nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 2, padding = (0,1)),
                        torch.nn.MaxPool2d(kernel_size = (2,4))
                        )

        self.classifier = torch.nn.Sequential(
                        torch.nn.Linear(128,128),
                        torch.nn.ReLU(),
                        )

        self.new_layer = torch.nn.Sequential(
                        torch.nn.Linear(128,256),
                        torch.nn.LeakyReLU(),
                        torch.nn.Linear(256,3),
                        torch.nn.LeakyReLU(),
                        torch.nn.Linear(3,1),
                        torch.nn.Sigmoid()
                        )

    def forward(self, x):
        x = self.features(x)
        x = torch.reshape(x,[-1,128])
        x = self.classifier(x)
        x = self.new_layer(x)
        x = 30.0 * x
        return x


def process_board(boards,rows = 6,columns = 7, mark = 1):

    boards = torch.FloatTensor(boards)
    my_marks = boards == mark
    my_marks = my_marks.float()

    opp_marks = torch.mul(boards != mark, boards >0)
    opp_marks = -1*opp_marks.float()
    processed_board = torch.reshape( torch.cat((my_marks, opp_marks)), (1,2,rows, columns) )

    return processed_board




#Convolutional neural net TD(lambda) Agent
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
        self.prev_state = None
        self.columns = None
        self.rows = None
        self.inarow = None
        self.z = None
        self.epsilon = None
        self.delta = None


    def agent_init(self, agent_info={}):
        """Setup for the agent called when the experiment first starts."""
        self.lambd = agent_info.get("lambda", 0.2)
        self.gamma = agent_info.get("gamma", 0.5)
        self.alpha = agent_info.get("alpha", 0.001)
        self.columns = agent_info.get("columns", 7)
        self.rows = agent_info.get("rows", 6)
        self.inarow = agent_info.get("inarow", 4)
        self.delta = 0.0
        self.epsilon = 0.0
        self.model = ConnectNet4()

        pretrained_dict = torch.load('board_classifier.pt')

        model_dict = self.model.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        self.model.load_state_dict(model_dict)
        for param in self.model.features.parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = False

        self.z = [torch.zeros(weights.shape, requires_grad=False) for weights in list(self.model.new_layer.parameters())]


    def agent_start_episode(self):

        x_init = np.zeros(6*7)
        self.model( process_board(x_init,rows = 6,columns = 7, mark = 1) ).backward()
        self.model.zero_grad()
        self.z = [torch.zeros(weights.shape, requires_grad=False) for weights in list(self.model.new_layer.parameters())]


    def select_action(self, state):
        """
        Selects an action using greedy approach. Picks the column that maximizes
        the probability of winning.
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
                possible_board = torch.reshape(torch.FloatTensor(test_drop(board, c)),(rows, columns))
                possible_features.append(possible_board)
                action_values.append( self.model( process_board(board,rows = 6,columns = 7, mark = mark) ).item() )



            action = possible_actions[argmax(action_values)]
            features = process_board(possible_features[argmax(action_values)], mark = mark)

        self.prev_state = features

        return action

    def agent_update(self, reward, state):
        """Updates the weights of the agent based off of the state of the board
        after the opponent makes a move.
        Args:
            state (Numpy array): the state observation from the
                environment's evn_start function.
            reward: float that is zero if the game is still being played or if the
                agent has lost and 30.0 if the game is finished and the agent won.
        """
        board = state['board']
        columns = self.columns
        rows = self.rows
        mark = state['mark']

        self.model.zero_grad()

        v_prev =  self.model( self.prev_state )

        v_prev.backward()


        with torch.no_grad():
            #get the new features

            current_state = process_board(board,rows = 6,columns = 7, mark = mark)
            current_v = self.model( current_state ).item()
            # get the parameters of the model
            parameters = list(self.model.new_layer.parameters())


            self.delta = reward + self.gamma * current_v - v_prev.item()

            for i, weights in enumerate(parameters):

                # z <- gamma * lambda * z + (grad w w.r.t P_t)
                self.z[i] = self.gamma * self.lambd * self.z[i] + weights.grad

                # w <- w + alpha * td_error * z
                weights += self.delta * self.alpha * self.z[i]


    def last_agent_update(self, reward):

        """Updates the weights of the model according to the TD(lambda) algorithm upon
        termination of the game (or epsilode)
        Args:
            reward: float that is 0 in the event of a loss or tie and 30.0 in the
            event of a win.
        """

        self.model.zero_grad()

        v_prev =  self.model( self.prev_state )

        v_prev.backward()


        with torch.no_grad():

            # get the parameters of the model
            parameters = list(self.model.new_layer.parameters())

            #print(v_prev)

            self.delta = reward - v_prev.item()
            #print(self.delta)
            for i, weights in enumerate(parameters):

                # z <- gamma * lambda * z + (grad w w.r.t P_t)
                self.z[i] = self.gamma * self.lambd * self.z[i] + weights.grad
                #print(self.delta * self.alpha *  self.z[i])
                # w <- w + alpha * td_error * z
                weights+= self.delta * self.alpha *  self.z[i]
