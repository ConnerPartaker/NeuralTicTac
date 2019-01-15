import numpy as np

ALPHA = .01

#function must take z*, derivative must take z
def logarithmic (x) : return 1.0/(1+np.exp(-x)) 
def logarithmicD(x) : return x*(1.0 - x)
def ReLu(x)         : return np.maximum(0, x)
def ReLuD(x)        : return np.sign(x)


class BaseLayer:

    def __init__(self):
        self.outputs = None
        self.inbox   = None

class DenseLayer(BaseLayer):

    def __init__(self, sigmoid = 'log'):
        super().__init__()
        self.inputL  = None
        self.weights = None
        self.partial = None
        self.sigdef = sigmoid
    
    def sigmoid(self, x): 
        if self.sigdef == 'log':  return logarithmic(x)
        if self.sigdef == 'ReLu': return ReLu(x)
        return 0
    
    def sigmoidD(self, x):
        if self.sigdef == 'log':  return logarithmicD(x)
        if self.sigdef == 'ReLu': return ReLuD(x)
        return 0
        
    def feedforward(self):
        self.outputs = self.sigmoid(np.dot(self.weights, self.inputL.outputs))
                            
    def part(self):         
        self.partial = self.sigmoidD(self.outputs)*self.inbox
    
    def backprop(self):
        self.part()
        self.inputL.inbox = np.dot(np.transpose(self.weights), self.partial)
        self.weights -= np.dot(np.outer(self.partial, self.inputL.outputs), ALPHA)

        
class EndLayer (DenseLayer):

    def __init__(self, sigmoid = 'log'):
        super().__init__(sigmoid)
        self.expect  = None
        
    def part(self):
        self.partial = 2*(self.outputs-self.expect)*self.sigmoidD(self.outputs)



         
class NeuralNet:

    def __init__(self, size=[3, 1], sigmoid = []):
        
        sigmoid += ['log']*(len(size)-1 - len(sigmoid))
        self.layers = [BaseLayer()] + [DenseLayer(s) for s in sigmoid[:-1]] + [EndLayer(sigmoid[-1])]
        
        # Create connections between layers, init weights matrix
        for i in range(1, len(size)):
            self.layers[i].inputL = self.layers[i-1]
            self.layers[i].weights = np.random.rand(size[i], size[i-1])
        
        self.worksheets = []
    
    def feedforward(self, injected):
        
        self.layers[0].outputs = injected
        for k in self.layers[1:]: k.feedforward()
        
        return self.layers[-1].outputs
    
    def backprop(self, expected):
        
        self.layers[-1].expect = expected
        for k in self.layers[:0:-1]: k.backprop()
                
    def adapt(self, x=1, worksheets = None):
        for _ in range(x):
            for i in worksheets:
                self.feedforward(i[0])
                self.backprop(i[1])