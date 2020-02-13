import numpy as np

class MLP(object):
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.l1 = Layer(input_dim     = input_dim,
                                   output_dim = hidden_dim,
                                   activation    = self.sigmoid,
                                   dactivation     = self.dsigmoid)
        self.l2 = Layer(input_dim     =hidden_dim,
                                   output_dim = output_dim,
                                   activation    = self.sigmoid,
                                   dactivation     = self.dsigmoid)
        self.layers = [self.l1, self.l2]
        
    def __call__(self, x):
        return self.forward(x)
        
    def forward(self, x):
        h = self.l1(x)
        y = self.l2(h)
        return y
    
    def sigmoid(self, x):
        return 1 / (1+np.exp(-x))
    
    def dsigmoid(self, x):
        return self.sigmoid(x) * (1- self.sigmoid(x))
    
    def compute_loss(self, t, y):
         return (-t * np.log(y) - (1-t) * np.log(1-y)).sum()
        
    def train_step(self, x, t):
        y = self(x)
        
        for i, layer in enumerate(self.layers[::-1]):
            if i == 0:
                delta = y-t
            else:
                delta = layer.backward(delta, W)
        
            dW, db = layer.compute_gradients(delta)
            layer.W = layer.W - 0.1 * dW
            layer.b = layer.b - 0.1 * db
        
            W = layer.W
        loss = self.compute_loss(t, y)
        return loss

class Layer(object):
    def __init__(self, input_dim, output_dim, activation, dactivation):
        self.W = np.random.normal(size=(input_dim, output_dim))
        self.b = np.zeros(output_dim)
        
        self.activation = activation
        self.dactivation = dactivation
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        self._input = x
        self._pre_activation = np.matmul(x, self.W) + self.b
        return self.activation(self._pre_activation)
    
    def backward(self, delta, W):
        delta = self.dactivation(self._pre_activation) * np.matmul(delta, W.T)
        return delta
    
    def compute_gradients(self, delta):
        dW = np.matmul(self._input.T, delta)
        db = np.matmul(np.ones(self._input.shape[0]), delta)
        return dW, db