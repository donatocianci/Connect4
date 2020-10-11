
import numpy as np
from random import choice
from ConnectX_env import ConnectX
from random import sample
import torch


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

def train(num_episodes, players):
    '''

    '''

    win_record = []

    for episode in range(num_episodes):

        agents = sample(players, 2)

        agents[0].agent_start_episode()
        agents[1].agent_start_episode()
        first_play = choice([1,2])

        game = ConnectX()
        game.start_game(mark = first_play)

        while game.termination != True:
            action = agents[game.state['mark']-1].select_action(game.state)
            reward, done, state = game.step(action)
            if done:
                agents[game.state['mark']-1].last_agent_update(reward)
                if reward >0:
                    win_record.append(game.state['mark'])
                else:
                    win_record.append(0)
            else:
                agents[game.state['mark']-1].agent_update(reward, state)
            #switch turns
            game.state['mark'] = 1 + game.state['mark']%2

    return win_record

def evaluate(num_episodes, agents):
    '''

    '''

    win_record = []

    for episode in range(num_episodes):
        agents[0].agent_start_episode()
        agents[1].agent_start_episode()
        first_play = choice([1,2])

        game = ConnectX()
        game.start_game(mark = first_play)

        while game.termination != True:
            action = agents[game.state['mark']-1].select_action(game.state)
            reward, done, state = game.step(action)
            if done:
                if reward >0:
                    win_record.append(game.state['mark'])
                else:
                    win_record.append(0)
            #switch turns
            game.state['mark'] = 1 + game.state['mark']%2

    return win_record

def generate_examples(num_episodes, agents,existing_database = set(), epsilon = 0.3):

    for episode in range(num_episodes):
        agents[0].agent_start_episode()
        agents[1].agent_start_episode()
        first_play = choice([1,2])

        game = ConnectX()
        game.start_game(mark = first_play)

        while game.termination != True:
            action = agents[game.state['mark']-1].select_action(game.state)
            reward, done, state = game.step(action)
            p = np.random.rand()
            if not done and p < epsilon:
                existing_database.add((tuple(state['board']), 0))
            if done:
                if reward >0:
                    existing_database.add((tuple(state['board']), state['mark']))
                else:
                    existing_database.add((tuple(state['board']), 0))

            #switch turns
            game.state['mark'] = 1 + game.state['mark']%2

    return existing_database


def watch_play(agents):
    '''

    '''

    first_play = choice([1,2])
    game = ConnectX()
    game.start_game(mark = first_play)

    while game.termination != True:
        action = agents[game.state['mark']-1].select_action(game.state)
        reward, done, state = game.step(action)
        game.render()


        game.state['mark'] = 1 + game.state['mark']%2
