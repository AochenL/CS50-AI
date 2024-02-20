"""
Tic Tac Toe Player
"""

from ftplib import MAXLINE
import math
from copy import deepcopy
NoneType = type(None)

X = "X"
O = "O"
EMPTY = None

def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    x_num = 0
    o_num = 0

    for i in range(3):
        for j in range(3):
            if board[i][j] == X:
                x_num += 1
            elif board[i][j] == O:
                o_num += 1
    
    if x_num <= o_num:
        return X
    return O

def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    actions = set()
    for i in range(3):
        for j in range(3):
            if board[i][j] == None:
                actions.add((i,j))
    
    return actions

def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    # check if action is a valid one
    
    if action == None:
        raise ValueError
    i = action[0]
    j = action[1]
    
    new_board = deepcopy(board)
    new_board[i][j] = player(board)
    return new_board

def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # check horizontally
    for i in range(3):
        if board[i][0] == board[i][1] and board[i][1] == board[i][2] and board[i][0]:
            return board[i][0]

    # check vertically
    for j in range(3):
        if board[0][j] == board[1][j] and board[1][j] == board[2][j] and board[0][j]:
            return board[0][j]

    # check diagonally 
    if board[0][0] == board[1][1] and board[1][1] == board[2][2] and board[1][1]:
        return board[1][1]
    if board[0][2] == board[1][1] and board[1][1] == board[2][0] and board[1][1]:
        return board[1][1]
    
    #if no winner, return None
    return None

def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    # someone win
    if winner(board):
        return True

    # all cells are taken
    num = 0
    for i in range(3):
        for j in range(3):
            if board[i][j] != None:
                num += 1
    if num == 9:
        return True

    # not over
    return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == X:
        return 1
    elif winner(board) == O:
        return -1
    return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    alpha = float('-inf') 
    beta = float('inf') 
    if player(board) == X:
        return max_value_action(board, alpha, beta)[1]
    else:
        
        return min_value_action(board, alpha, beta)[1]

    
def max_value_action(board, alpha, beta):
    """
    Returns a tuple: (value, action)
    alpha is the minimum score that the maximizing player is assured of
    beta is the maximum score that the minimizing player is assured of
    For alpha-beta code is referenced from Wikipedia
    """
    if terminal(board):
        return (utility(board), None)
    v = float('-inf')
    move = None
    for action in actions(board):
        new_move_value = min_value_action(result(board, action), alpha, beta)[0]
        if new_move_value > v:
            v = new_move_value
            move = action
        if v >= beta:
            break
        alpha = max(alpha, v)

    return (v, move)
    
def min_value_action(board, alpha, beta):
    if terminal(board):
        return (utility(board), None)
    v = float('inf')
    move = None
    
    for action in actions(board):
        new_move_value = max_value_action(result(board, action), alpha, beta)[0]
        if new_move_value < v:
            v = new_move_value
            move = action
        if v <= alpha:
            break
        beta = min(beta, v)
        
    return (v, move) 
      
    