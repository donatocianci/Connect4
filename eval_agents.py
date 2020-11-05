from random import choice
from RL_utils import check_winning_move

class BaseAgent:

    def __init__(self):

        self.columns = None
        self.rows = None

    def agent_init(self, agent_info={}):

        self.columns = agent_info.get("columns", 7)
        self.rows = agent_info.get("rows", 6)

    def select_action(self, state):
        """
        Selects a random action.
        Args:
        state - list of the board state
        Returns:
        an integer between 0 and 6 for which column to play a particular piece
        """

        board = state['board']
        columns = self.columns
        rows = self.rows
        mark = state['mark']

        return choice([c for c in range(columns) if board[c] == 0])

    def agent_update(self, reward, state):
        pass

    def last_agent_update(self, reward):
        pass

    def agent_start_episode(self):
        pass


class McGoo(BaseAgent):

    def select_action(self, state):

        board = state['board']
        columns = self.columns
        rows = self.rows
        mark = state['mark']

        valid_moves = [col for col in range(columns) if board[col] == 0]
        winning_moves = [col for col in valid_moves if check_winning_move(board, rows, columns, col, mark) == True]
        if winning_moves:
            return choice(winning_moves)
        op_winning_moves = [col for col in valid_moves if
                             check_winning_move(board, rows, columns, col, mark%2+1) == True]
        if op_winning_moves:
            return choice(op_winning_moves)
        return choice(valid_moves)





class StepPlay(BaseAgent):
    """
    Agent used for playing against a particular agent.
    """

    def select_action(self, state):

        board = state['board']
        columns = self.columns
        rows = self.rows
        mark = state['mark']

        valid_moves = [c for c in range(columns) if board[c] == 0]
        finish_turn = False

        while finish_turn != True:
            piece = input("Choose a column to place a "+str(mark)+" piece : \n")
            if int(piece) in valid_moves:
                finish_turn = True
            else:
                print("Not a possible move. Choose again. \n")

        return int(piece)
