
import numpy as np


class Optimizer(object):

    def __init__(self, param=None, eps=.01, eta=10, regl=.01, momentum=.5, optim="tempo",  *args, **kwargs):
        super(Optimizer, self).__init__(*args, **kwargs)

        self._param = param
        self._dparam = np.zeros_like(self._param)
        self._param_t = np.array(
            [np.zeros_like(self._param), np.zeros_like(self._param)])
        self._momentum = momentum
        self._eta = eta

        self._e = 1e-8  # -7
        self._eps = eps
        self._lambda = regl
        self._t = 0
        self._decay = 1e5
        self._mem = np.zeros_like(self._param)
        self._v = np.zeros_like(self._param)
        self._bm, self._b1, self._b2 = .5, .9, .999
        self._dic_op = {"sgd": self.descent, "rms": self.rmsprop, "adagard": self.adagard, "tempo": self.temporal,
                        "momentum": self.momentum, "adam": self.adam, "adamx": self.ada_max}
        self.optimizer = self._dic_op[optim]

    def delay_no_norm(self, eps):
        self._t += 1
        eps = eps/(1+self._t/self._decay)
        return eps

    def norm_no_delay(self):
        self._param = self._param/(self._param**2 + 1)

    def delay_norm(self, eps):
        cof = 1/(1+self._t/self._decay)
        self._dparam = self._dparam/(self._dparam**2 + 1)
        self._t = (self._t + 1)
        eps = eps*cof
        return eps

    def temporal(self):
        self._param = self._param - self._eta*self._dparam
        self._param = self._param + self._momentum * \
            np.diff(self._param_t, axis=0)[0]
        self._param_t[1] = self._param_t[0].copy()
        self._param_t[0] = self._param


    def momentum(self):
        b1, eps = self._b1, self._eps
        self._v = b1 * self._v + eps * self._dparam
        self._param = self._param - self._v

    def adagard(self):
        eps, e = self._eps, self._e
        self._mem = self._mem + self._dparam*self._dparam
        q = self._dparam / (np.sqrt(self._mem) + e)
        self._param = self._param - eps * q

    def descent(self):
        eps = self._eps
        lam = self._lambda
        self._dparam = self._dparam + lam*self._param
        self._param = self._param - eps * self._dparam

    def rmsprop(self):
        b1, eps, e = self._b1, self._eps, self._e
        self._mem = (b1 * self._mem + (1 - b1) * self._dparam*self._dparam)
        self._param = self._param - eps*self._dparam/(np.sqrt(self._mem) + e)

    def ada_max(self):
        b1, b2, eps, e = self._b1, self._b2, self._eps, self._e
        self._mem = b1 * self._mem + (1 - b1) * self._dparam
        self._v = np.maximum(b2 * self._v + e,  np.abs(self._dparam))
        q = self._mem / self._v
        self._param = self._param - eps * q

    def adam(self):
        self._t += 1
        b1, b2, e, eps = self._b1, self._b2, self._e, self._eps
        t = self._t
        self._mem = b1 * self._mem + (1 - b1) * self._dparam
        self._v = b2 * self._v + (1 - b2) * self._dparam*self._dparam

        mt = self._mem / (1-np.power(b1, t))
        vt = self._v / (1-np.power(b2, t))
        self._param = self._param - eps * mt / (np.sqrt(vt) + e)

    def update_params(self, Y, dY):
        self._param = Y
        self._dparam = dY
        self.optimizer()
        # self.clip_param(self._param, -1., 1.)

    def clip_param(self, M, mn, mx):
        np.clip(M, mn, mx, out=M)

    def reset_dparams(self):
        self._dparam = np.zeros_like(self._param)

    def get_param(self):
        return self._param

    def set_param(self, P):
        self._param = P

    def get_dparam(self):
        return self._dparam

    def set_dparam(self, dP):
        self._dparam = dP

    def get_momentum(self):
        return self._momentum

    def set_momentum(self, m):
        self._momentum = m
