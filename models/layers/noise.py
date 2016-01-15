import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams


eps = 1e-8
zero = np.float32(0.)
one = np.float32(1.)


def noise_conditions(params, index, noiseWhere):
    doNoise = (('addNoise' in params.rglrz) or (index == 0 and 'inputNoise' in params.rglrz)) and params.noiseWhere == noiseWhere
    return doNoise           

        
def noiseup(input, splitPoint, noiz, noiseType, params, index, rstream):
    
    ''' Additive and multiplicative Gaussian Noise
    
    '''

    if 'inputNoise' in params.rglrzPerMap or 'addNoise' in params.rglrzPerMap:
        noiz = noiz.dimshuffle('x', 0, 'x', 'x') 

    if   noiseType == 'multi1':                
        input = T.set_subtensor(input[:splitPoint],
                input[:splitPoint]*(1. + noiz*rstream.normal(input[:splitPoint].shape, dtype=theano.config.floatX)))
    elif noiseType == 'multi0':                
        input = T.set_subtensor(input[:splitPoint],
                input[:splitPoint]*(noiz*rstream.normal(input[:splitPoint].shape, dtype=theano.config.floatX)))
    else:
        input = T.set_subtensor(input[:splitPoint],
                input[:splitPoint] + noiz*rstream.normal(input[:splitPoint].shape, dtype=theano.config.floatX))    
    return input


def dropout(input, splitPoint, params, rglrzParam, nIn, rstream):

    ''' Dropout noise
    
    '''    

    if 'dropOut' in params.rglrz:
        mask = rstream.binomial(n=1, p=(1.-rglrzParam['dropOut']), size=input[:splitPoint].shape, dtype=theano.config.floatX)
        input = T.set_subtensor(input[:splitPoint], input[:splitPoint] * mask)
        input = T.set_subtensor(input[splitPoint:], input[splitPoint:] * (1-rglrzParam['dropOut']))
    elif 'dropOutB' in params.rglrz:
        mask = rstream.binomial(n=1, p=(1.-rglrzParam['dropOutB']), size=(nIn,), dtype=theano.config.floatX)
        input = T.set_subtensor(input[:splitPoint], input[:splitPoint] * mask)
        input = T.set_subtensor(input[splitPoint:], input[splitPoint:] * (1-rglrzParam['dropOutB']))

