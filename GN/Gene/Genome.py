'''

@author: dan
'''



from copy import deepcopy
from .GPGraphNode import GPGraphNode
from .GPConstNode import GPConstNode
from .GPVarNode import GPVarNode
from .GeneExpression import GeneExpression
from .GOperationsDef import GOperationsDef
import numpy as np


class Genome:

    def __init__(self, ops={"opx": 1, "depth": 3, "nvars": 2, "pvc": .8, "pf": 1, "cross": .4, "mut": .2, "mrand": 5.}):

        self._GP_FunDef = GOperationsDef()
        self._GeneExpression = GeneExpression()
        self._OpsIndex = ops["opx"]
        self._maxDepth = ops["depth"]
        self._numVars = ops["nvars"]
        self._mutation = ops["mut"]  # .5
        self._cross = ops["cross"]  # 0.5
        self._TorCprob = ops["pvc"]  # 0.6
        self._FuncProb = ops["pf"]
        self._mrand = ops["mrand"]
        self._Operations = self._GP_FunDef.getFuncSlots()
        self._nGraph = None
        self._flist = self._Operations[self._OpsIndex]  # self.get_Operations()

    def createGraph(self, depth=None):
        if depth == None:
            depth = np.random.randint(1, self._maxDepth+1)

        if np.random.random() < self._FuncProb and depth > 0:
            funct = np.random.choice(self._flist)
            children = [self.createGraph(depth-1)
                        for i in range(funct._paramsNums)]
            return GPGraphNode(funct, children)

        if np.random.random() < self._TorCprob:
            # return GPVarNode( randint( 0, self._numVars - 1 ) ) # math random
            # numpy add 1 toupper limit
            return GPVarNode(np.random.randint(0, self._numVars))

        else:
            return GPConstNode(np.random.uniform(-self._mrand, self._mrand))

    def display(self):
        return self._GeneExpression.convertToExpression(self._nGraph)

    def evaluate(self, _input):
        return self._nGraph.evaluate(_input)

    def create(self):
        self._nGraph = self.createGraph()

    def crossOver(self, otherNeuron):
        self._nGraph = self.crossoverGraph(self._nGraph, otherNeuron._nGraph)

    def mutate(self):
        self._nGraph = self.mutateGraph(self._nGraph)

    def mutateGraph(self, t):
        if np.random.random() < self._mutation:
            return self.createGraph()
        else:
            nGraph = deepcopy(t)
            if isinstance(t, GPGraphNode):
                nGraph._children = [self.mutateGraph(
                    child) for child in t._children]

            return nGraph

    def crossoverGraph(self, t1, t2, top=True):

        if (np.random.random() < self._cross) and not top:
            return deepcopy(t2)
        else:
            aGraph = deepcopy(t1)
            if isinstance(t1, GPGraphNode) and isinstance(t2, GPGraphNode):
                aGraph._children = [self.crossoverGraph(
                    c, np.random.choice(t2._children), False) for c in t1._children]
            return aGraph

    def setOpIndex(self, idx):
        self._OpsIndex = idx

    def setMut(self, mut):
        self._mutation = mut

    def setBreed(self, breed):
        self._cross = breed

    def setNumVars(self, nv):
        self._numVars = nv

    def getNumVars(self):
        return self._numVars

    def get_Operations(self):
        return self._Operations[self._OpsIndex]

    def set_Operations(self, index):
        self._OpsIndex = index

    def getDepth(self):
        return self._maxDepth

    def setDepth(self, dpt):
        self._maxDepth = dpt

    def setFprob(self, fpt):
        self._FuncProb = fpt

    def setVprob(self, fpt):
        self._TorCprob = fpt

    def __repr__(self):
        return self.repr()

    def __str__(self):
        return self.display()
