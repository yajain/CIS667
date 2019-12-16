'''
Implements a simple CLI
'''
from __future__ import print_function
import sys
import re
from random import randint

from omegabot import OmegaBot

import numpy as np
from othello import OthelloBoard, BSIZE, WHITE, BLACK, print_board

g_brd = OthelloBoard()
g_bot = OmegaBot()
print('Compiling model ... ', end='')
sys.stdout.flush()
g_bot.build_model()
print('Done')

with open('model.pkl', 'rb') as f:
    print('Loaded parameters from model.pkl')
    g_bot.load_params(f)

def rng_player(brd_):
    ar = np.clip(g_brd.movable, 0,1).flat
    idxs = [i for i,v in enumerate(ar) if v != 0]
    y,x = divmod(idxs[randint(0,len(idxs)-1)], BSIZE)
    print('rng> %s%d'%(chr(x+65),y+1))
    return x,y

def omega_player(brd_):
    print('omega> ', end='')
    sys.stdout.flush()
    x,y = g_bot.gen_move_minmax(brd_)
    print('%s%d'%(chr(x+65),y+1))
    return x,y

#None->human player
g_ai_players = {WHITE: omega_player, BLACK: None}

def move_cb(retval_):
    global g_brd
    if retval_==0:
        return
    elif retval_ == -1:
        print('Invalid move!')
    else:
        print_board(g_brd)
        print(['Black Wins!', 'White Wins!', 'Draw!'][retval_-1])
        input('Press ENTER to continue')
        g_brd.reset()

def play():
    global g_brd, g_ai_players
    while True:
        print_board(g_brd)
        if g_ai_players[g_brd.cur_plr] is not None:
            move_cb(g_brd.move(*(g_ai_players[g_brd.cur_plr](g_brd))))
            continue
        try:
            cmd = input('human> ')
        except EOFError:
            break

        if cmd in ['q', 'exit']:
            break
        if re.match('[a-zA-Z][0-9]', cmd):
            x = (ord(cmd[0])|32)-97
            y = int(cmd[1])-1
            move_cb(g_brd.move(x,y))
        elif cmd =='x':
            g_ai_players[BLACK], g_ai_players[WHITE] = g_ai_players[WHITE], g_ai_players[BLACK]
        elif cmd in ['new', 'n']:
            g_brd.reset()
        elif cmd in ['help', 'h']:
            print('new\tStart a new game.')
            print('x\tExchange player, enter this at start to play white.')
            print('exit\tSelf explanatory.')
            print('help\tShow this message.' )
        elif cmd == '':
            continue
        else:
            print('Unknown command, type "help" for list of commands')

if __name__=='__main__': play()
