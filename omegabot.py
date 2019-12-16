from __future__ import print_function

from math import sqrt, pi
from random import randint
from six.moves import cPickle as pickle

import numpy as np
import theano as th
import theano.tensor as T
from optimize import AdamSGD, VanillaSGD

from othello import OthelloBoard, BSIZE

th.config.floatX = 'float32'

g_params_di = {}

def _using_params(params_di_):
    global g_params_di
    g_params_di = params_di_

def get_sharedv(name_, shape_, init_range_=None, dtype_=th.config.floatX):
    global g_params_di
    if name_ in g_params_di:
        #TODO: add shape/dtype check?
        return g_params_di[name_]
    if init_range_ is None:
        v = th.shared(
            np.zeros(shape_,dtype=dtype_),
            name=name_
        )
    else:
        v = th.shared(
            np.asarray(np.random.uniform(
                *init_range_,
                size=shape_),
                dtype=dtype_),
            name=name_
        )
    g_params_di[name_] = v
    return v

def lyr_conv(name_, s_x_, idim_, odim_, fsize_=(3,3)):
    name_conv_W = '%s_w'%name_
    name_conv_B = '%s_b'%name_
    init_range = 0.3/sqrt(idim_*fsize_[0]*fsize_[1]+odim_)
    v_conv_W = get_sharedv(
        name_=name_conv_W,
        shape_=(odim_,idim_,*fsize_),
        init_range_=(-init_range,init_range)
    )
    v_conv_B = get_sharedv(
        name_=name_conv_B,
        shape_=(odim_,),
    )
    get_sharedv(name_conv_W, (odim_, idim_, *fsize_), (-init_range,init_range))
    get_sharedv(name_conv_B, (odim_, idim_, *fsize_), (-init_range,init_range))
    return T.nnet.conv2d(
        s_x_, v_conv_W,
        filter_shape=(odim_, idim_, *fsize_),
        border_mode = 'half'
    )+v_conv_B.dimshuffle('x',0,'x','x')

def lyr_linear(name_, x_, idim_, odim_):
    name_W = name_+'_w'
    name_B = name_+'_b'
    init_range = 2.23/sqrt(idim_+odim_)
    v_W = get_sharedv(name_W, shape_=(idim_, odim_), init_range_=(-init_range,init_range))
    v_B = get_sharedv(name_B, shape_=(odim_,), init_range_=(-init_range,init_range))
    return T.dot(x_, v_W) + v_B
    # return T.dot(x_, v_W)

def lyr_gru(name_, s_x_, s_state_, idim_, sdim_, lyr_linear_, axis_=-1):
    in_gate = T.nnet.sigmoid(lyr_linear_(name_+'_igate',T.join(axis_, s_x_, s_state_), idim_+sdim_, idim_))
    rec_gate = lyr_linear
    s_gated_x = s_x_ * in_gate
    s_interp_lin, s_state_tp1_lin = T.split(lyr_linear_(name_+'_main', T.join(axis_,s_gated_x, s_state_), idim_+sdim_, sdim_*2), [sdim_]*2, 2, axis_)
    s_interp = T.nnet.sigmoid(s_interp_lin)
    return T.tanh(s_state_tp1_lin)*s_interp + s_state_*(1.-s_interp)

def lyr_gru_nogate(name_, s_x_, s_state_, idim_, sdim_, lyr_linear_, axis_=-1):
    s_interp_lin, s_state_tp1_lin = T.split(lyr_linear_(name_+'_main', T.join(axis_,s_x_, s_state_), idim_+sdim_, sdim_*2), [sdim_]*2, 2, axis_)
    s_interp = T.nnet.sigmoid(s_interp_lin)
    return T.tanh(s_state_tp1_lin)*s_interp + s_state_*(1.-s_interp)

def lyr_lstm(name_, s_x_, s_state_, idim_, sdim_, odim_, lyr_linear_, axis_=-1):
    raise NotImplementedError()

def _board_to_nnfmt(brd_li_):
    '''
    Calculates a numpy array with value 0.0/1.0 for NNs
    The returned array has shape 4*BSIZE*BSIZE
    the 1st channel is full of 1s, this helps CNN to recognize border (since it's going to be padded by 0s)
    the 2nd channel contains pieces of current player
    the 3rd channel contains peices of opponent
    the 4nd channel contains movable places
    Args:
        brd_li_: List of OthelloBoard instances
    Returns: dst_ or new array if dst_ is None
    '''
    floatx = th.config.floatX
    dst_ = np.zeros((len(brd_li_), 4,BSIZE,BSIZE), dtype=floatx)
    for i in range(len(brd_li_)):
        brd = brd_li_[i]
        dst_[i,0,:,:] = 1.
        dst_[i,1] = (brd.board==brd.cur_plr).astype(floatx)
        dst_[i,2] = (brd.board==(brd.cur_plr^3)).astype(floatx)
        dst_[i,3] = np.clip(brd.movable, 0,1).astype(floatx)
    return dst_

class OmegaBot(object):
    def __init__(self):
        self.optimizer = AdamSGD()

    def build_model(self):
        self.params_di = {}
        _using_params(self.params_di)

        #observation/action/reward sequence
        s_obsv = T.tensor4()
        s_movs = T.ivector()
        s_rwrd = T.vector()

        s_state = T.nnet.relu(lyr_conv('conv0', s_obsv, 4, 32))
        s_state_li, _ = th.scan(
            lambda _s,_o: lyr_gru('gru', _o, _s, idim_=4, sdim_=32, lyr_linear_=lyr_conv, axis_=1),
            outputs_info=[s_state],
            non_sequences=[s_obsv],
            n_steps = 8
        )
        s_state_top = s_state_li[-1]
        top_dims = 32*BSIZE*BSIZE
        s_f0 = T.nnet.relu(lyr_linear('mlp0', s_state_top.flatten(ndim=2), top_dims, 512))
        s_f1 = T.nnet.relu(lyr_linear('mlp1', s_f0, 512, 256))
        s_bsize = T.shape(s_obsv)[0]
        s_qval = T.tanh(lyr_linear('mlp2', s_f1, 256, 64) )-2*(1.-T.reshape(s_obsv[:,3], (s_bsize, BSIZE*BSIZE)))
        s_val = T.sum(T.extra_ops.to_one_hot(s_movs, BSIZE*BSIZE)*s_qval, axis=1)
        # s_val = T.max(s_qval, axis=1)
        s_loss = T.mean(T.sqr(s_val - s_rwrd))
        self.fn_eval = th.function([s_obsv], s_qval)
        self.fn_train = self.optimizer.compile([s_obsv, s_movs, s_rwrd], s_loss, self.params_di.values(), fetches_=s_loss)

    def save_params(self,file_):
        pickle.dump({k:np.asarray(v.get_value()) for k,v in self.params_di.items()}, file_)

    def load_params(self,file_):
        for k,v in pickle.load(file_).items():
            self.params_di[k].set_value(v)

    def gen_move(self,brd_):
        inp = _board_to_nnfmt([brd_])
        qval = self.fn_eval(inp)
        m = np.argmax(qval)
        y,x = divmod(m, BSIZE)
        return x,y

    def _eval_minmax(self, brd_, depth_):
        '''
        Very simple minmax without pruning
        Returns: value, (x,y)
        '''
        if depth_==0:
            qval = self.eval_board(brd_)
            yx = np.argmax(qval)
            y,x = divmod(yx, BSIZE)
            return qval[yx], (x,y)
        moves = [(lambda _:(_[1],_[0]))(divmod(yx,BSIZE)) for yx in range(BSIZE*BSIZE) if brd_.movable.flat[yx]!=0]
        branches = []
        for m in moves:
            sub_brd = OthelloBoard()
            res = brd_.move(*m, sub_brd)
            if res==0:
                branches.append(-self._eval_minmax(sub_brd, depth_-1)[0])
            elif res==brd_.cur_plr:
                branches.append(1.)
            elif res^3==brd_.cur_plr:
                branches.append(-1.)
        imove = np.argmax(branches)
        return branches[imove], moves[imove]

    def gen_move_minmax(self, brd_, depth_=3):
        return self._eval_minmax(brd_, depth_)[1]

    def eval_board(self,brd_):
        qval = self.fn_eval(_board_to_nnfmt([brd_]))
        return qval[0]

    def train_rl(self, v_brd_, v_moves_, v_rwrd_):
        return self.optimizer.fit(
            _board_to_nnfmt(v_brd_),
            np.asarray(v_moves_, np.int32),

            np.asarray(v_rwrd_,dtype=th.config.floatX))
