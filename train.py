import sys
from random import randint, uniform

import numpy as np
import numba as nb

from othello import OthelloBoard, WHITE, BLACK, BSIZE, print_board
from omegabot import OmegaBot

def _gen_random_move(brd_):
    ar = np.clip(brd_.movable, 0,1).flat
    idxs = [i for i,v in enumerate(ar) if v != 0]
    y,x = divmod(idxs[randint(0,len(idxs)-1)], BSIZE)
    return x,y

def train(bot_, nsess_, nips_, batch_size_=128, expl_=0.5):
    '''
    train omegabot using self-play
    Args:
        bot_: OmegaBot instance
        nsess_:
            total number of training sessions
            bot will updated between sessions
        nips_: iterations per session
        batch_size_: size of minibatch to be fed into NN
        expl_: epsilon-greedy exploration rate
    '''
    coinflip = lambda : randint(0,65535)>32767
    brd = OthelloBoard()
    try:
        for isess in range(nsess_):
            brd_li = []
            rwrd_li = []
            moves_li = []
            brd.reset()
            plr = WHITE if coinflip() else BLACK
            just_new_game = True
            i=0
            while i<nips_:
                if plr == brd.cur_plr:
                    brd_li.append(OthelloBoard(brd))
                    qval = bot_.eval_board(brd)
                    if uniform(0.,1.)<expl_:
                        yx =(lambda _:_[0]+_[1]*BSIZE)(_gen_random_move(brd))
                    else:
                        yx = np.argmax(qval)
                    moves_li.append(yx)
                    y,x = divmod(yx,BSIZE)
                    res = brd.move(x,y)
                    if just_new_game:
                        just_new_game = False
                        continue
                    rwrd_li.append(qval[yx])
                    # rwrd_li.append(0.)
                    i+=1
                    if i==nips_: break
                    if res == plr:
                        i+=1
                        rwrd_li.append(1.)
                        brd.reset()
                        plr = WHITE if coinflip() else BLACK
                        just_new_game = True
                    elif res == plr^3:
                        i+=1
                        rwrd_li.append(-1.)
                        brd.reset()
                        plr = WHITE if coinflip() else BLACK
                        just_new_game = True
                    elif res == 3:
                        i+=1
                        rwrd_li.append(0.)
                        brd.reset()
                        plr = WHITE if coinflip() else BLACK
                        just_new_game = True
                    elif res == -1:
                        i+=1
                        rwrd_li.append(-1.)
                        brd.reset()
                        plr = WHITE if coinflip() else BLACK
                        just_new_game = True
                else:
                    if uniform(0.,1.)<expl_ or just_new_game:
                        res = brd.move(*_gen_random_move(brd))
                    else:
                        res = brd.move(*bot_.gen_move(brd))
                    if res == plr:
                        i+=1
                        rwrd_li.append(1.)
                        brd.reset()
                        plr = WHITE if coinflip() else BLACK
                        just_new_game = True
                    elif res == plr^3:
                        i+=1
                        rwrd_li.append(-1.)
                        brd.reset()
                        plr = WHITE if coinflip() else BLACK
                        just_new_game = True
                    elif res == 3:
                        i+=1
                        rwrd_li.append(0.)
                        brd.reset()
                        plr = WHITE if coinflip() else BLACK
                        just_new_game = True
                    elif res == -1:
                        i+=1
                        rwrd_li.append(-1.)
                        brd.reset()
                        plr = WHITE if coinflip() else BLACK
                        just_new_game = True
            loss = 0.
            for ibatch in range(0,nips_,batch_size_):
                loss += bot_.train_rl(
                    brd_li[ibatch:ibatch+batch_size_],
                    moves_li[ibatch:ibatch+batch_size_],
                    rwrd_li[ibatch:ibatch+batch_size_]
                )
            loss /= (nips_//batch_size_)
            print('Sess %d/%d %d iters: loss %f'%(isess+1,nsess_,nips_,loss))
    except KeyboardInterrupt:
        print('User hit CTRL-C, abort')
        raise

if __name__ == '__main__':
    bot = OmegaBot()
    print('Compiling ... ', end='')
    sys.stdout.flush()
    bot.build_model()
    print('Done')

    try:
        with open('model.pkl', 'rb') as f:
            bot.load_params(f)
    except FileNotFoundError:
        pass

    try:
        train(bot, 10000, 1024, expl_=0.04)
    except KeyboardInterrupt:
        with open('model.pkl', 'wb') as f:
            bot.save_params(f)

    try:
        with open('model.pkl', 'wb') as f:
            bot.save_params(f)
    except:
        print('Failed to save a model')
