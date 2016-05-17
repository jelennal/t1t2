import numpy as np
import theano.tensor as T

eps = np.float32(1e-8)
zero = np.float32(0.)
one = np.float32(1.)
convNonLin = 'relu'

def relu(input):
    output = T.maximum(0., input)
    return output
    
def activation(input, key):
    ''' 
        Defining various activation functions.    
    '''
    identity = lambda x: x    
    activFun = {'lin':  identity,
                'relu': relu,
                'elu':  T.nnet.elu,
                'tanh': T.tanh, 
                'sig':  T.nnet.sigmoid, 
                'softmax': T.nnet.softmax}[key]
    
    return activFun(input)   


def weight_multiplier(nIn, nOut, key):    
    ''' 
        Initial range of values for weights, given diffrent activation functions.    
    '''    
    weightMultiplier = {'lin':  np.sqrt(1./(nIn+nOut)), 
                        'relu': np.sqrt(1./(nIn+nOut))*np.sqrt(12), 
                        'elu':  np.sqrt(1./(nIn+nOut))*np.sqrt(12), 
                        'tanh': np.sqrt(1./(nIn+nOut))*np.sqrt(6.),
                        'sig':  np.sqrt(1./(nIn+nOut))*np.sqrt(6.)/4, 
                        'softmax': 1e-5}[key]

    return weightMultiplier       
    
    
    