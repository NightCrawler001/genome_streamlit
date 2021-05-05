'''

@author: dan
'''


import numpy as np
from copy import deepcopy
from .NBase import NBase


class NRegressor(NBase):

    def predict(self, anet, inp):
        inout = inp[:]
        for lay in anet:
            inout = lay.evaluate(inout)

        # res = self.sigmoid(inout[0]) # using activation and set genes f con var becomes ANN
        res = inout[0]
        return res

    def _score(self, x, y):
        p = self.predict(self._net, x)
        # loss = (p - y)**2 # mse
        # loss = np.abs(np.log(np.cosh(p - y))) # log
        # loss = np.abs(p - y) # abs
        delta = 10 # hoss
        loss = np.where(np.abs(y-p) < delta , 0.5*((y-p)**2), delta*np.abs(y - p) - 0.5*(delta**2))
        return loss

    def getYhat(self, nx=False, multi=False):
        X = self._idatas["X"]
        if nx:
            X = self._idatas["nX"]
        yhat = np.array([self.predict(self._best_net, x) for x in X])

        return X, yhat


    def test_me(self):
        
        nX = self._idatas["nX"]
        nY = self._idatas["nY"]
        bnet = self.get_best_net()
        nYhat = np.array([self.predict(bnet, x) for x in nX])
        y_real = nY

        print(nYhat, "=====PRED====== BEST")
        print(y_real, "=====REAL====== TEST")
        self.display(bnet)
