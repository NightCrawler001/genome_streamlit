from abc import abstractmethod, ABC

import numpy as np
from copy import deepcopy
from .Layer import Layer
from datetime import datetime
from random import choice

class NBase(ABC):
    def __init__(self, dim=[], datas=None, nr_opts={}, g_opts={}, **kwargs):
        self._dim = dim  # [inp_nvars,l1,l2,lout]
        self._size = len(dim)-1
        self._idatas = datas
        self._net = np.empty(shape=(self._size,), dtype=np.object)
        self._loading = False
        self._Terminate = False
        self._before = True
        self._cur_score = np.inf
        self._best_score = np.inf
        self._best_net = None
        self._prev_nets = []

        self._max_data_batch = self._idatas["X"].shape[0]-1
        self._batch_size = min(
            g_opts["bsize"], int(self._max_data_batch))  # 60
        # max( 100 , int(self._max_data_batch/self._batch_size))  # 500 # REG 5000 should be X / lm to zap almost all intervals..
        self._max_epoch = g_opts["mxepoch"]
        # 5 #  int( .25*self._maxEpoch / (self._max_data_batch/self._batch_size) ) +1
        self._batch_update = g_opts["bupdate"]
        self._max_tries = g_opts["mxtries"]  # 5
        self._history = g_opts["history"]  #
        self._mode = g_opts["mode"]
        self._fract = g_opts["fraction"]
        self._nOpts = nr_opts
        self._cur_epoch = 1


    @abstractmethod
    def test_me(self):
        raise NotImplementedError('This is an abstract class')

    @abstractmethod
    def _score(self,x,y):
        raise NotImplementedError('This is an abstract class')

    def create(self):
        le = self._size + 1
        for i in range(1, le):
            n = self._dim[i]
            self._nOpts["nvars"] = self._dim[i-1]
            layer = Layer(num_nodes=n, n_opts=self._nOpts)
            layer.create()
            self._net[i-1] = layer

    def get_score(self, XY):
        scr_f = self._score
        scores = np.array([scr_f(x, y) for x, y in XY])
        # self._cur_score = scores.mean()
        self._cur_score = scores.sum()

    def shift_idatas(self, st, en):
        if self._cur_epoch % self._batch_update == 0:
            st += self._batch_size + np.random.randint(0, 10)
            en = st + self._batch_size
            if en > self._max_data_batch:
                st = 0
                en = st + self._batch_size

        return st, en

    def fast_cross_mutate(self, XY):
        o_net = choice(self._prev_nets)
        ix = np.random.randint(0, self._size)
        ox = np.random.randint(0, self._size)
        ml = self._net[ix]
        ol = o_net[ox]  # ol = o_net[ix]
        iy = np.random.randint(0, ml._num_nodes)
        oy = np.random.randint(0, ol._num_nodes)
        ml._nodes[iy].mutate()
        self.check_score(XY)
        ml._nodes[iy].crossOver(ol._nodes[oy])
        self.check_score(XY)

    def cross_mutate(self, XY):
        for ix in range(self._size):
            o_net = choice(self._prev_nets)
            ml = self._net[ix]
            ol = o_net[np.random.randint(0, self._size)]  # ol = o_net[ix]
            iy = np.random.randint(0, ml._num_nodes)
            oy = np.random.randint(0, ol._num_nodes)
            ml._nodes[iy].mutate()
            self.check_score(XY)
            ml._nodes[iy].crossOver(ol._nodes[oy])
            self.check_score(XY)

    def check_score(self, XY):
        self.get_score(XY)
        if self._cur_score < self._best_score:
            self.update_score()

    def darwiny(self, XY):
        t = 0
        while (t <= self._max_tries):
            self.fast_cross_mutate(XY)
            # self.cross_mutate(XY)
            t += 1

    def update_score(self):
        self._best_score = self._cur_score
        self._best_net = deepcopy(self._net)
        self._prev_nets.append(deepcopy(self._best_net))
        self._prev_nets = self._prev_nets[-self._history:]

    def create_update(self):
        self.create()
        if not self._best_net or self._best_net is None:
            self._best_net = deepcopy(self._net)

    def print_info(self, st):
        elapsed_time = datetime.now() - st
        e,m = self.get_cur_epoc()-1 , self.get_max_epoc()
        c,b = self.get_cur_score() , self.get_best_score()
        print(f'[{elapsed_time}] ->  epochs: {e}/{m} -- {np.round(100*e/m,2)}%  -- score: {c}/{b} ')

    def getNextBatch(self, X, Y, st, en):
        st, en = self.shift_idatas(st, en)
        XX = X[st:en]
        YY = Y[st:en]
        XY = list(zip(XX, YY))
        return st, en, XY

    def first_populate(self, XY):
        self.create_update()
        self.update_score()
        self.darwiny(XY)

    def train(self):

        dic_data = self._idatas
        X = dic_data["X"]
        Y = dic_data["Y"]
        st = 0
        en = st + self._batch_size
        start_time = datetime.now()
        XY = list(zip(X, Y))
        self.first_populate(XY)

        while (self._cur_epoch <= self._max_epoch and not self._Terminate):

            # if self._cur_epoch % 50 == 0:
            #     self.print_info(start_time)

            if self._cur_epoch < self._fract*self._max_epoch:
                self.create_update()
            else:
                self._net = deepcopy(self._best_net)

            st, en, XY = self.getNextBatch(X, Y, st, en)
            self.darwiny(XY)

            self._cur_epoch += 1

            if self._best_score <= 0.01:
                break
            
            self.print_info(start_time)
            self.test_me()

        self._net = deepcopy(self._best_net)
        self._loading = False

    def display(self, a_net):
        i = 1
        n = ""
        for l in a_net:
            n += "\n------ layer: " + str(i) + " ----------\n"
            n += str(l)
            i += 1
        print(n)


    def display2(self, a_net):
        i = 1
        n = ""
        for l in a_net:
            n += "\n------ layer: " + str(i) + " ----------\n"
            n += l.__repr__()
            i += 1
        print(n)

    def softmax(self, x):
        e_x = np.exp((x - np.max(x)))
        #e_x = np.exp(x)
        out = e_x / e_x.sum()
        return out

    def sigmoid(self, x):
        return .5 * (1 + np.tanh(.5 * x))

    def relu(self, x):
        return np.maximum(x, .001*x, x)

    def setLoading(self, l):
        self._loading = l

    def getLoading(self):
        return self._loading

    def getDatas(self):
        return self._idatas

    def getMode(self):
        return self._mode

    def setTerminate(self, am):
        self._Terminate = am

    def get_cur_epoc(self):
        return self._cur_epoch

    def get_max_epoc(self):
        return self._max_epoch

    def times_max_epoc(self, m):
        self._max_epoch *= m

    def get_best_net(self):
        return self._best_net

    def get_best_score(self):
        return self._best_score

    def get_cur_score(self):
        return self._cur_score

    def setBefore(self,b):
        self._before = b

    def getBefore(self):
        return self._before


    def set_prevs(self, prev):
        self._prev_nets = prev

    def set_net(self, anet):
        self._net = anet
