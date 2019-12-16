import sys
import numpy as np
import numba as nb

BSIZE = 8
BLACK = 1
WHITE = 2
BOARD_CHR = chr(0x25e6)+chr(0x25cb)+chr(0x25cf)
board_t = nb.int32[:,:]

@nb.jit(nb.int32(board_t, board_t, nb.int32))
def _update_movable(brd_, movable_, plr_):
    ''' Update movable steps on a board
    Returns:
        0: nothing special
        1: the player has no available moves
    '''
    movable_[:,:] = 0
    oppo = plr_^3
    cant_move = 1
    for xy in range(BSIZE*BSIZE):
        y,x = divmod(xy, BSIZE)
        if brd_[y,x] != plr_: continue

        if x+1<BSIZE:
            if brd_[y,x+1] == oppo:
                for ix in range(x+2,BSIZE):
                    if brd_[y,ix] == oppo: continue
                    if brd_[y,ix] == 0:
                        movable_[y,ix] |= 1
                        cant_move = 0
                    break

        if x-1>0:
            if brd_[y,x-1] == oppo:
                for ix in range(x-2,-1,-1):
                    if brd_[y,ix] == oppo: continue
                    if brd_[y,ix] == 0:
                        movable_[y,ix] |= 2
                        cant_move = 0
                    break

        if y+1<BSIZE:
            if brd_[y+1,x] == oppo:
                for iy in range(y+2,BSIZE):
                    if brd_[iy,x] == oppo: continue
                    if brd_[iy,x] == 0:
                        movable_[iy,x] |= 4
                        cant_move = 0
                    break

        if y-1>0:
            if brd_[y-1,x] == oppo:
                for iy in range(y-2,-1,-1):
                    if brd_[iy,x] == oppo: continue
                    if brd_[iy,x] == 0:
                        movable_[iy,x] |= 8
                        cant_move = 0
                    break

        if x+1<BSIZE and y+1<BSIZE:
            if brd_[y+1,x+1] == oppo:
                for pp in range(2,BSIZE-max(x,y)):
                    if brd_[y+pp,x+pp] == oppo: continue
                    if brd_[y+pp,x+pp] == 0:
                        movable_[y+pp,x+pp] |= 16
                        cant_move = 0
                    break

        if x-1>0 and y-1>0:
            if brd_[y-1,x-1] == oppo:
                for pp in range(2,1+min(x,y)):
                    if brd_[y-pp,x-pp] == oppo: continue
                    if brd_[y-pp,x-pp] == 0:
                        movable_[y-pp,x-pp] |=32
                        cant_move = 0
                    break

        if x+1<BSIZE and y-1>0:
            if brd_[y-1,x+1] == oppo:
                for pp in range(2,1+min(BSIZE-x-1,y)):
                    if brd_[y-pp,x+pp] == oppo: continue
                    if brd_[y-pp,x+pp] == 0:
                        movable_[y-pp,x+pp] |=64
                        cant_move = 0
                    break

        if x-1>0 and y+1<BSIZE:
            if brd_[y+1,x-1] == oppo:
                for pp in range(2,1+min(x,BSIZE-y-1)):
                    if brd_[y+pp,x-pp] == oppo: continue
                    if brd_[y+pp,x-pp] == 0:
                        movable_[y+pp,x-pp] |= 128
                        cant_move = 0
                    break
    return cant_move


@nb.jit(nb.int32(board_t, board_t, nb.int32, nb.int32, nb.int32, nb.int32))
def _make_move(dst_, src_, movable_, x_, y_, plr_):
    '''
    Returns:
        0: nothing special
        1: invalid move
    '''
    dst_[:,:] = src_
    if not (0<=x_<BSIZE and 0<=y_<BSIZE): return 1
    if movable_==0: return 1

    if movable_&2:
        dst_[y_,x_+1] = plr_
        for ix in range(x_+2,BSIZE):
            if dst_[y_,ix] == plr_: break
            dst_[y_,ix] = plr_
    if movable_&1:
        dst_[y_,x_-1] = plr_
        for ix in range(x_-2,-1,-1):
            if dst_[y_,ix] == plr_: break
            dst_[y_,ix] = plr_
    if movable_&8:
        dst_[y_+1,x_] = plr_
        for iy in range(y_+2,BSIZE):
            if dst_[iy,x_] == plr_: break
            dst_[iy,x_] = plr_
    if movable_&4:
        dst_[y_-1,x_] = plr_
        for iy in range(y_-2,-1,-1):
            if dst_[iy,x_] == plr_: break
            dst_[iy,x_] = plr_
    if movable_&32:
        dst_[y_+1,x_+1] = plr_
        for pp in range(2,BSIZE-max(x_,y_)):
            if dst_[y_+pp, x_+pp] == plr_: break
            dst_[y_+pp, x_+pp] = plr_
    if movable_&16:
        dst_[y_-1,x_-1] = plr_
        for pp in range(2,1+min(x_,y_)):
            if dst_[y_-pp, x_-pp] == plr_: break
            dst_[y_-pp, x_-pp] = plr_
    if movable_&128:
        dst_[y_-1,x_+1] = plr_
        for pp in range(2,1+min(BSIZE-x_,y_+1)):
            if dst_[y_-pp, x_+pp] == plr_: break
            dst_[y_-pp, x_+pp] = plr_
    if movable_&64:
        dst_[y_+1,x_-1] = plr_
        for pp in range(2,1+min(BSIZE-y_,x_+1)):
            if dst_[y_+pp, x_-pp] == plr_: break
            dst_[y_+pp, x_-pp] = plr_
    dst_[y_,x_] = plr_
    return 0


class Board(object):
    #TODO: abstractize some stuff here
    pass

class OthelloBoard(Board):
    __slots__ = ('board', 'movable', 'cur_plr', 'moves')
    ''''''
    def __init__(self, src_ = None):
        if src_ is None:
            self.reset()
        else:
            self.board = src_.board.copy()
            self.movable = src_.movable.copy()
            self.cur_plr = src_.cur_plr
            self.moves = src_.moves

    def reset(self):
            self.board = np.zeros((BSIZE,BSIZE), dtype='int32')
            self.movable = np.zeros((BSIZE, BSIZE), dtype='int32')
            self.cur_plr = 1
            self.moves = 0
            hsize = BSIZE//2
            self.board[hsize-1:hsize+1, hsize-1:hsize+1] = [[WHITE,BLACK],[BLACK,WHITE]]
            self._update_movable()

    def _update_movable(self):
        return _update_movable(self.board, self.movable, self.cur_plr)

    def move(self, x_, y_, dst_=None):
        '''
        Make a move, returns non-zero value if endgame event happens
        Args:
            x_, y_: coordinate to move
            dst_: destination board, self if None
        Returns:
            -1 if invalid move
            0 if nothing special
            1 if black wins
            2 if white wins
            3 if draw
        '''
        if dst_ is not None:
            dst_.board[:,:] = self.board[:,:]
            dst_.movable[:,:] = self.movable[:,:]
            dst_.cur_plr = self.cur_plr
        else: dst_ = self
        if 1==_make_move(dst_.board, self.board, self.movable[y_,x_], x_, y_, self.cur_plr):
            return -1
        dst_.cur_plr^=3
        if 1==dst_._update_movable():
            dst_.cur_plr^=3
            if 1==dst_._update_movable():
                black_pcs = np.count_nonzero(dst_.board==1)
                white_pcs = np.count_nonzero(dst_.board==2)
                if black_pcs>white_pcs: return BLACK
                elif black_pcs<white_pcs: return WHITE
                else: return 3
        return 0

def print_board(brd_):
    print(BOARD_CHR[brd_.cur_plr]+' A B C D E F G H')
    for y in range(BSIZE):
        sys.stdout.write(str(y+1) + ' ')
        for x in range(BSIZE):
            sys.stdout.write((chr(0x2a2f) if brd_.movable[y,x] else BOARD_CHR[brd_.board[y,x]])+' ')
        sys.stdout.write('\n')
    black_pcs = np.count_nonzero(brd_.board==1)
    white_pcs = np.count_nonzero(brd_.board==2)
    print( ' '*4+(BOARD_CHR[BLACK]+' %2d  %2d '+BOARD_CHR[WHITE])%(black_pcs, white_pcs) )
    sys.stdout.flush()
