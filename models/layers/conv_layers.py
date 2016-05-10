import numpy as np
import theano
import theano.tensor as T

import theano.tensor.nnet.conv as nnconv
from theano.tensor.signal import pool#downsample

from models.layers.shared import t1_shared, t2_shared
from models.layers.activations import activation
from models.layers.noise import noise_conditions, noiseup, dropout
from models.layers.batchnorm import bn_shared, bn_layer

eps = 1e-8
zero = np.float32(0.)
one = np.float32(1.)
convNonLin = 'relu'


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
            W, b, a = t1_shared(params=params, rng=rng, index=index, nIn=nIn, nOut=nOut, 
                                outFilters=outFilters, filterShape=filterShape) 

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

        # add gauss noise before affine transform                 
        if noise_conditions(params, index, 'type0'):
            input = noiseup(input, splitPoint, noiz, params.noiseT1, params, index, rstream)

        # convolution
        convOut = nnconv.conv2d(input, self.W, subsample = stride, 
                                     border_mode = params.convLayers[index].border)

        # batch normalization & scale+shift   
        if params.batchNorm and params.convLayers[index].bn:
            convOut, updateBN = bn_layer(convOut, self.a, self.b, self.normParam, params, splitPoint, graph)
            self.updateBN = updateBN 
        else:
            convOut += self.b.dimshuffle('x', 0, 'x', 'x') 

        # add gauss noise before nonlinearity         
        if noise_conditions(params, index, 'type1'): 
            convOut = noiseup(convOut, splitPoint, noiz, params.noiseT1, params, index, rstream)        
        # nonlinearity
        self.output = activation(convOut, convNonLin)
     
                              
class pool_layer(object):
   def __init__(self, rstream, input, params, index, splitPoint, graph,
               poolShape, inFilters, outFilters, stride, ignore_border = False, 
               b=None, a=None, normParam=None, rglrzParam=None):

        ''' 
            Pooling layer + BN + noise 
        '''        
        # noise
        self.paramsT2 = []
        if 'addNoise' in params.rglrz and params.convLayers[index].noise:

            if rglrzParam is None:
                self.rglrzParam = {}
                tempValue = params.rglrzInitial['addNoise'][index]            
                tempParam = np.asarray(tempValue, dtype=theano.config.floatX)
                noizParam = theano.shared(value=tempParam, name='%s_%d' % ('addNoise', index), borrow=True)
                self.rglrzParam['addNoise']=noizParam
            if params.useT2 and 'addNoise' in params.rglrzTrain:
                self.paramsT2 = [noizParam]
                
            input = noiseup(input, splitPoint, noizParam, params.noiseT1, params, index, rstream)

        #  pooling          
        self.output = pool.pool_2d(input, ds = poolShape, st = stride, 
                                            ignore_border = ignore_border, mode = 'max')                                                                                                

        # batch normalization
        if params.batchNorm and params.convLayers[index].bn:    
            
            _, b, a = t1_shared(params=params, rng=0, index=index, nIn=0, nOut=0, 
                                outFilters=outFilters, filterShape=0, defineW=0) 

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
                                                                                   
                                                                                   
class average_layer(object):
    def __init__(self, rstream, input, params, index, splitPoint, graph,
                 poolShape, inFilters, outFilters, stride, ignore_border = False, 
                 b=None, a=None, normParam=None, rglrzParam=None):
        ''' 
            Averaging layer + BN + noise 
        '''                
        # noise   
        self.paramsT2 = []
        if 'addNoise' in params.rglrz and params.convLayers[index].noise:
            if rglrzParam is None:
                self.rglrzParam = {}
                tempValue = params.rglrzInitial['addNoise'][index]            
                tempParam = np.asarray(tempValue, dtype=theano.config.floatX)
                noizParam = theano.shared(value=tempParam, name='%s_%d' % ('addNoise', index), borrow=True)
                self.rglrzParam['addNoise']=noizParam
            if params.useT2 and 'addNoise' in params.rglrzTrain:
                self.paramsT2 = [noizParam]
            #self.output = noiseup(self.output, splitPoint, noizParam, params.noiseT1, params, index, rstream)
            input = noiseup(input, splitPoint, noizParam, params.noiseT1, params, index, rstream)

        # averaging
        self.output = pool.pool_2d(input, ds = poolShape, st = stride, 
                                             ignore_border = ignore_border, mode = 'average_exc_pad')

        # if batch normalization                                             
        if params.batchNorm and params.convLayers[index].bn:            
            
            _, b, a = t1_shared(params=params, rng=0, index=index, nIn=0, nOut=0, 
                                outFilters=outFilters, filterShape=0, defineW=0) 

            self.b = b; self.a = a    
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
            self.output = activation(self.output, 'softmax')
                


        