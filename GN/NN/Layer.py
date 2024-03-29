'''

@author: dan
'''


import numpy as np
from copy import deepcopy
from GN.Gene.Genome import Genome


class Layer(object):
    def __init__(self, num_nodes, n_opts, **kwargs):
        self._num_nodes = num_nodes
        self._nOpts = n_opts
        self._nodes = np.empty(shape=(self._num_nodes,), dtype=np.object)

    def create(self):
        #self._nodes  = np.array([Neuron(ops=self._nOpts) for x in xrange(self._num_nodes) ], dtype=np.object)
        for i in range(self._num_nodes):
            n_ero = Genome(ops=self._nOpts)
            n_ero.create()
            self._nodes[i] = n_ero

    def evaluate(self, inp):
        EV = np.array([nd.evaluate(inp) for nd in self._nodes])
        return EV

    def __repr__(self):
        nodes = " "
        for nd in self._nodes:
            nodes += f" {nd.__repr__()} \n"

        return nodes

    def __str__(self):
        nodes = " "
        for nd in self._nodes:
            nodes += f" {nd.__str__()} \n"

        return nodes
