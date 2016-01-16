import numpy as np
import theano
import theano.tensor as T

import theano.tensor.nnet.conv as nnconv
from theano.tensor.signal import downsample
from theano.tensor.shared_randomstreams import RandomStreams

from models.layers.shared import t1_shared, t2_shared
from models.layers.noise import noise_conditions, noiseup, dropout
from models.layers.batchnorm import bn_shared, bn_layer

relu = lambda x: T.maximum(0, x)
identity = lambda x: x
eps = 1e-8
zero = np.float32(0.)
one = np.float32(1.)
convNonLin = 'relu'


def softmax(x):
    e_x = T.exp(x - x.max(axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)
    
def activation(input, key):
    relu = lambda x: T.maximum(0, x)    
    activFun = {'lin': identity, 'tanh': T.tanh, 'relu': relu, 
                'sig': T.nnet.sigmoid, 'softmax': softmax}[key]
    return activFun(input)   


class conv_layer(object):
   def __init__(self, rng, rstream, index, splitPoint, input, params, globalParams, graph,
                filterShape, inFilters, outFilters, stride, 
                W=None, b=None, a=None, rglrzParam=None, normParam=None, normWindow=None):
                    
        ''' Class defining a convolutional layer.
            # imageShape  ::  (0 batch size, 1 # in feature maps,         2 in image height, 3 in image width)  
            # filterShape ::  (0 # out feature maps, 1 # in feature maps, 2 filter height, 3 filter width)
        
        # Arguments:
        
        '''                                        
        # defining filter dimensions           
        filterDim = (filterShape[0], filterShape[1])     
        filterShape = (outFilters, inFilters, filterShape[0], filterShape[1]) 
        nIn = inFilters*filterDim[0]*filterDim[1] 
        nOut = outFilters*filterDim[0]*filterDim[1]
        updateBN = []
        
        ''' 
            Defining shared variables: T1, T2, BN
        '''
                
        # defining shared T1 params
        if W is None:
            W, b, a = t1_shared(params, rng, index, nIn, nOut, outFilters, filterShape) 
        self.W = W; self.b = b; self.a = a    
        if params.batchNorm and not params.aFix:
            self.paramsT1 = [W, b, a]
        else:    
            self.paramsT1 = [W, b]
            
        # defining shared T2 params      
        self.paramsT2 = []
        if rglrzParam is None:
            rglrzParam = t2_shared(params, globalParams, index, inFilters, outFilters, filterShape) 
                                                                                      
        self.rglrzParam = rglrzParam               
        if params.useT2:
            for rglrz in params.rglrzTrain:
                if (rglrz not in params.rglrzPerNetwork) and (rglrz not in params.rglrzPerNetwork1):
                        self.paramsT2 += [rglrzParam[rglrz]] # if trained, put param here

        #  defining shared BN params
        if params.batchNorm and params.convLayers[index].bn:          
            if normParam is None: 
                normParam = bn_shared(params, outFilters, index)                       
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

        # add gauss noise after nonlinearity                  
        self.input = input
        if noise_conditions(params, index, 'type0'):
            input = noiseup(input, splitPoint, noiz, params.noiseT1, params, index, rstream)


        # convolution
        self.convOut = nnconv.conv2d(input, self.W, subsample = stride, border_mode = 'valid')
#                                     image_shape=self.input.shape, filter_shape = filterShape,                                     

        # batch normalization & scale+shift   
        if params.batchNorm and params.convLayers[index].bn:
            self.convOut, updateBN = bn_layer(self.convOut, self.a, self.b, self.normParam, params, splitPoint, graph)
            self.updateBN = updateBN 
        else:
            self.convOut += self.b.dimshuffle('x', 0, 'x', 'x') 

        # add gauss noise before nonlinearity         
        if noise_conditions(params, index, 'type1'): 
            self.convOut = noiseup(self.convOut, splitPoint, noiz, params.noiseT1, params, index, rstream)        
        # nonlinearity
        self.output = activation(self.convOut, convNonLin)
     
                              
class pool_layer(object):
   def __init__(self, rstream, input, params, index, splitPoint, graph,
               poolShape, inFilters, outFilters, stride, ignore_border = False, 
               b=None, a=None, normParam=None, noizParam=None):

        ''' 
            Pooling layer + BN + noise 
        '''        
        #  pooling          
        self.input = input
        self.output = downsample.max_pool_2d(input, ds = poolShape, st = stride, 
                                            ignore_border = ignore_border, mode = 'max')                                                                                                
        # if batch normalization
        if params.batchNorm and params.convLayers[index].bn:    
            
            _, b, a = t1_shared(params, 0, index, 0, 0, outFilters, 0, 0) 
            self.b = b; self.a = a     
            if params.batchNorm and not params.aFix:
                self.paramsT1 = [b, a]
            else:    
                self.paramsT1 = [b]

            if normParam is None: 
                normParam = bn_shared(params, outFilters, index)                                 
            self.normParam = normParam         
            self.paramsBN = [normParam['mean'], normParam['var'], normParam['iter']]
            self.output, updateBN = bn_layer(self.output, self.a, self.b, self.normParam, params, splitPoint, graph)
            self.updateBN = updateBN

        # noise
        if 'addNoise' in params.rglrz and params.convLayers[index].noise:
            if noizParam is None: 
                tempValue = params.rglrzInitial['addNoise'][index]            
                tempParam = np.asarray(tempValue, dtype=theano.config.floatX)
                noizParam = theano.shared(value=tempParam, name='%s_%d' % ('addNoise', index), borrow=True)
                self.noizParam=noizParam
            if params.useT2 and 'addNoise' in params.rglrzTrain:
                self.paramsT2 = [noizParam]
            self.output = noiseup(self.output, splitPoint, noizParam, params.noiseT1, params, index, rstream)
                                                                                   
                                                                                   
class average_layer(object):
    def __init__(self, input, params, index, splitPoint, graph,
                 poolShape, inFilters, outFilters, stride, ignore_border = False, 
                 b=None, a=None, normParam=None, noizParam=None):
        ''' 
            Averaging layer + BN + noise 
        '''        
        # averaging
        self.input = input
        self.output = downsample.max_pool_2d(input, ds = poolShape, st = stride, 
                                             ignore_border = ignore_border, mode = 'average_exc_pad')

        # if batch normalization                                             
        if params.batchNorm and params.convLayers[index].bn:
            
            _, b, a = t1_shared(params, 0, index, 0, 0, outFilters, 0, 0) 
            self.b = b; self.a = a    
            if params.batchNorm and not params.aFix:
                self.paramsT1 = [b, a]
            else:    
                self.paramsT1 = [b]
                        
            if normParam is None: 
                normParam = bn_shared(params, outFilters, index)                                                  
            self.normParam = normParam         
            self.paramsBN = [normParam['mean'], normParam['var'], normParam['iter']]
            self.output, updateBN = bn_layer(self.output, self.a, self.b, self.normParam, params, splitPoint, graph)
            self.updateBN = updateBN 
 
        # flattening and softmax 
        self.output = T.flatten(self.output, outdim = 2)                                     
        if params.convLayers[index].type == 'average+softmax':
            self.output = softmax(self.output)

                


        