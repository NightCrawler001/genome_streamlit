'''

@author: dan
'''


import numpy as np
from copy import deepcopy
from .Layer import Layer
from .NBase import NBase


class NClassifier(NBase):

    def predict(self, anet, inp):
        inout = inp[:]
        for lay in anet:
            inout = lay.evaluate(inout)
            #inout = self.sigmoid(inout)
            #inout = np.tanh(inout)

        res = inout
        res = self.softmax(res)  # with  log score
        return res

    def _score(self, x, y):
        p = self.predict(self._net, x)
        
        # pm = np.argmax(p)
        # loss = np.log(np.cosh(pm - y))
        # loss += np.power(np.maximum(0,1-pm*y) , 2 )
        # pequals = np.equal(pm, y)
        # loss = (pequals == False)
        
        loss = -np.log(p[y])   
        
        return loss

    def get_boundaries(self):
        X = self._idatas["X"]
        mx = np.min(X)
        Mx = np.max(X)
        P = []
        XX = []
        for x1 in np.linspace(mx, Mx, num=200):
            for x2 in np.linspace(mx, Mx, num=200):
                x = np.array([x1, x2])
                XX.append(x)
                p = self.predict(self._best_net, x)
                P.append(p)
        YY = np.argmax(np.array(P), axis=1)
        XX = np.array(XX)

        return XX, YY

    def getYhat(self, nx=False, multi=False):
        X = self._idatas["X"]
        if nx:
            X = self._idatas["nX"]
        
        bnet = self.get_best_net()
        p = np.array([self.predict(bnet, x) for x in X])

        if multi:
            tp = np.argpartition(-p, 2)  # np.argmax(p, 1)
            #y_labels = tp[:,[0,2]]
            yhat = tp[:, :2]
        else:
            yhat = np.argmax(p, axis=1)

        return X, yhat

    def test_me(self):
        nX = self._idatas["nX"]
        nY = self._idatas["nY"]
        bnet = self.get_best_net()
        p = np.array([self.predict(bnet, x) for x in nX])
        y_real = nY
        #cm = self.confusion_matrix(yhats, y_real)
        #print cm
        print(np.argmax(p, axis=1), "=====PRED====== BEST")
        print(y_real, "=====REAL====== TEST")
        # self.display(self._best_net)



