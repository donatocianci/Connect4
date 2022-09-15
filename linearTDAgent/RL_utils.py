
import numpy as np
from random import choice
from random import sample
from sklearn.preprocessing import PolynomialFeatures


def argmax(q_values):
    top = float("-inf")
    ties = []

    for i in range(len(q_values)):
        if q_values[i] > top:
            top = q_values[i]
            ties = []

        if q_values[i] == top:
            ties.append(i)

    return np.random.choice(ties)
"""
def coarse_code_board(board, mark, rows, cols):

    board_grid = np.reshape(board, (rows, cols))

    my_marks = board_grid == mark
    opp_marks = np.multiply(board_grid != mark, board_grid > 0)

    my_X = np.hstack( (np.sum(my_marks,axis = 1)/rows , np.sum(my_marks, axis = 0)/cols) )
    opp_X = np.hstack( (np.sum(opp_marks,axis = 1)/rows , np.sum(opp_marks, axis = 0)/cols) )

    right_diags = []
    for d in range(-rows+1,cols):
        right_diags.append( np.average(np.diag(my_marks,d)) )

    left_diags = []
    for d in range(cols-1,-rows,-1):
        left_diags.append( np.average(np.diag(np.flip(my_marks, axis = 1),d)) )

    my_diags = np.hstack( (right_diags, left_diags) )
    my_X = np.hstack( (my_X,right_diags, left_diags) )

    right_diags = []
    for d in range(-rows+1,cols):
        right_diags.append( np.average(np.diag(opp_marks,d)) )

    left_diags = []
    for d in range(cols-1,-rows,-1):
        left_diags.append( np.average(np.diag(np.flip(opp_marks, axis = 1),d)) )

    opp_diags = np.hstack( (right_diags, left_diags) )
    opp_X = np.hstack( (opp_X,right_diags, left_diags) )

    X = np.hstack( (my_X,opp_X) )

    return X
"""
def coarse_code_board(board, mark, rows, cols):
    poly = PolynomialFeatures(degree=4, interaction_only=True, include_bias =False)
    board_grid = np.reshape(board, (rows, cols))

    my_marks = board_grid == mark
    opp_marks = np.multiply(board_grid != mark, board_grid > 0)
    my_feats = []
    for row in range(rows - 3):
        for col in range(cols - 3):
            
            sub_board = 1.0*my_marks[row:row+4,col:col+4] - 1.0*opp_marks[row:row+4,col:col+4]
            left_diag = np.diag(sub_board)
            right_diag = np.diag(np.flip(sub_board, axis = 1))
            columns = np.transpose(sub_board)
            sub_board_feats = np.vstack( (sub_board, left_diag, right_diag, columns) )
            
            sub_board_interactions = poly.fit_transform(sub_board_feats)
            my_feats.append(sub_board_interactions)
            
    features = np.hstack(my_feats).flatten()

    

    return features
