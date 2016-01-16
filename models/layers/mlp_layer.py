import numpy as np
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import theano

from models.layers.shared import t1_shared, t2_shared
from models.layers.noise import noise_conditions, noiseup, dropout
from models.layers.batchnorm import bn_shared, bn_layer

relu = lambda x: T.maximum(0, x)
identity = lambda x: x
eps = 1e-8
zero = np.float32(0.)
one = np.float32(1.)

def softmax(x):
    e_x = T.exp(x - x.max(axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)
    
def activation(input, key):
    relu = lambda x: T.maximum(0, x)    
    activFun = {'lin': identity, 'tanh': T.tanh, 'relu': relu, 
                'sig': T.nnet.sigmoid, 'softmax': softmax}[key]
    return activFun(input)   
    

class mlp_layer(object):
    def __init__(self, rng, rstream, index, splitPoint, input, params, globalParams, graph,# not layerwise params
                 W=None, b=None, a=None, rglrzParam=None, normParam=None, normWindow=None):                          # looper be share by the network  

        ''' Class defining a fully connected layer.
                                        
        '''
        if params.model == 'convnet':            
            nonLin = 'softmax'
            nIn = 10
            nOut = 10
        else:
            nonLin = params.activation[index]
            nIn = params.nHidden[index-1]
            nOut = params.nHidden[index]
            

        # defining shared T1 params
        if W is None:
            W, b, a = t1_shared(params, rng, index, nIn, nOut, nOut, 0)         
        self.W = W; self.b = b; self.a = a
        
        if params.batchNorm and not params.aFix and nonLin != 'softmax':
            self.paramsT1 = [W, b, a]
        else:    
            self.paramsT1 = [W, b]

        # defining shared T2 params
        self.paramsT2 = []
        if rglrzParam is None:
            rglrzParam = t2_shared(params, globalParams, index, nIn, nOut, 0) 

        self.rglrzParam = rglrzParam        
        self.paramsT2 = []
        if params.useT2:
            for rglrz in params.rglrzTrain:
                if (rglrz not in params.rglrzPerNetwork) and (rglrz not in params.rglrzPerNetwork1):
                    if rglrz != 'addNoise' or nonLin != 'softmax':
                        self.paramsT2 += [rglrzParam[rglrz]] # if trained, put param here


        # defining shared BN params
        if normParam is None and params.batchNorm and nonLin != 'softmax':                     
            normParam = bn_shared(params, nOut, index)     
            self.normParam = normParam         
            self.paramsBN = [normParam['mean'], normParam['var'], normParam['iter']]


        # noise
        if (index == 0 and 'inputNoise' in rglrzParam.keys()):
            noiz = self.rglrzParam['inputNoise']
        elif 'addNoise' in rglrzParam.keys():

            noiz = self.rglrzParam['addNoise']
            
            
        ''' 
            Input transformations: convolution, BN, noise, nonlinearity 
        '''


        # add gauss noise to input                     
        self.input = input
        if noise_conditions(params, index, 'type0'):
            input = noiseup(input, splitPoint, noiz, params.noiseT1, params, index, rstream)
            
        # dropout    
        if 'dropOut' in params.rglrz or 'dropOutB' in params.rglrz:
            input = dropout(input, splitPoint, params, self.rglrzParam, nIn, rstream)

        # affine transform    
        inputLin = T.dot(input, self.W)      
        
        # batchnorm transform
        if params.batchNorm and nonLin != 'softmax':
            inputLin, updateBN = bn_layer(inputLin, self.a, self.b, self.normParam, params, splitPoint, graph)
            self.updateBN = updateBN 
        else:
            inputLin += self.b
                
        # noise before nonlinearity
        if noise_conditions(params, index, 'type1'):
            inputLin = noiseup(inputLin, splitPoint, noiz, params.noiseT1, params, index, rstream)

        # nonlinearity  
        self.output = activation(inputLin, nonLin)
  


                   
        # ----------------------------------------------------------- PENALTY
#        # difference between Lmax trainable or not
#        if 'LmaxCutoff' in params.rglrz:
#            b = self.rglrzParam['LmaxCutoff'] # cutoff
#            a = self.rglrzParam['LmaxSlope'] # slope
#            tempNorm = T.sqrt(T.sum(T.sqr(self.W), axis=0))
#            self.LmaxW = T.sum(a*T.sqr(T.maximum(tempNorm - b, 0)))
#
#        if 'LmaxHard' in params.rglrz:
#            self.penaltyMaxParams = {self.W: self.rglrzParam['LmaxHard']}
#        elif 'LmaxHard' not in params.rglrzTrain:
#            self.penaltyMaxParams = {self.W: None}








