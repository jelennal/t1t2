import numpy as np
import theano.tensor as T

eps = 1e-8
zero = np.float32(0.)
one = np.float32(1.)
convNonLin = 'relu'


def softmax(x):
    e_x = T.exp(x - x.max(axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

    
def activation(input, key):
    relu = lambda x: T.maximum(0, x)    
    identity = lambda x: x    
    activFun = {'lin': identity, 'tanh': T.tanh, 'relu': relu, 
                'sig': T.nnet.sigmoid, 'softmax': softmax}[key]
    return activFun(input)   


def weight_multiplier(nIn, nOut, key):    

    ''' Initial range of values for weights, given diffrent activation functions    

    '''    

    weightMultiplier = {'lin': np.sqrt(1./nIn), 'tanh': np.sqrt(1./(nIn+nOut))*np.sqrt(6.),
                        'relu': np.sqrt(1./nIn), 'sig': np.sqrt(1./(nIn+nOut))*np.sqrt(6.)/4, 
                        'elu': np.sqrt(1./nIn), 'softmax': 1e-5}[key]
    return weightMultiplier       