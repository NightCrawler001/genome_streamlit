'''

@author: ben
'''


from .GPNode import GPNode


class GPVarNode(GPNode):

    def __init__(self, index):
        self._paramIndex = index

    def evaluate(self, _input):
        return _input[self._paramIndex]

    def display(self, indent=0):
        print('%sX%d' % (' ' * indent, self._paramIndex))

    def __repr__(self):
        return ' x' + str(self._paramIndex) + ' '

    def __str__(self):
        return ' x' + str(self._paramIndex) + ' '
    def nodes(self):
        return ['x' + str(self._paramIndex)]
