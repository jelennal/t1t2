import numpy as np
import theano
import theano.tensor as T

import theano.tensor.nnet.bn as bn
from theano.ifelse import ifelse

eps = 1e-8
zero = np.float32(0.)
one = np.float32(1.)


def bn_shared(params, outFilters, index):    

    ''' Setup BN shared variables.    

    '''
    normParam = {}       
    template = np.ones((outFilters,), dtype=theano.config.floatX)
    normParam['mean'] = theano.shared(value=0.*template, name='mean_%d' % (index), borrow=True)
    normParam['var'] = theano.shared(value=1.*template, name='var_%d' % (index), borrow=True)                                
    normParam['iter'] = theano.shared(np.float32(1.), name='iter')                 

    return normParam

def bn_layer(input, a, b, normParam, params, splitPoint, graph):

    ''' Apply BN.    

    # graph = 0 : T1 eval, T2 eval with MA, BN up with MA 
    # graph = 1 : T1 eval, T2 eval, no BN updates 
    # graph = 2 : T1 eval, T2 eval with T1BN, BN updated to T1BN
    # graph = 3 : T1 eval with BN, T2 eval with ?, BN updated to T1BN - for "proper" evaluation  

    '''
    minAlpha = params.movingAvMin
    iterStep = params.movingAvStep      
    alpha = ifelse(T.ge(graph, 2), one, T.maximum(minAlpha, 1./normParam['iter']))           
    alpha = ifelse(T.eq(graph, 1), zero, alpha)            
        
    # compute mean & variance    
    if params.model == 'convnet':
        mean1 = ifelse(T.le(splitPoint, 1), normParam['mean'], T.mean(input[:splitPoint], axis = (0, 2, 3) ))
        var1 = ifelse(T.le(splitPoint, 1), normParam['var'], T.var(input[:splitPoint], axis = (0, 2, 3) ))
    else:
        mean1 = ifelse(T.le(splitPoint, 1), normParam['mean'], T.mean(input[:splitPoint], axis = 0 ))
        var1 = ifelse(T.le(splitPoint, 1), normParam['var'], T.var(input[:splitPoint], axis = 0 ))
        
    # if T1 evaluated on the estimate, but its own average is saved            
    mean1 = ifelse(T.eq(graph, 3), normParam['mean'], mean1)
    var1 = ifelse(T.eq(graph, 3), normParam['var'], var1)         

    # moving average as a proxi for validation model 
    mean2 = (1-alpha)*normParam['mean'] + alpha*mean1 
    var2 = (1-alpha)*normParam['var'] + alpha*var1   

    # apply transformation: 
    # on the side of T1 stream, use T1 stats; on the side of T2 stream, use running average of T1 stats
    std1 = T.sqrt(var1 + eps); std2 = T.sqrt(var2 + eps)                
    if params.model == 'convnet':
        input = T.set_subtensor(input[:splitPoint], bn.batch_normalization(input[:splitPoint], 
                                a.dimshuffle('x', 0, 'x', 'x'), b.dimshuffle('x', 0, 'x', 'x'), 
                                mean1.dimshuffle('x', 0, 'x', 'x'), std1.dimshuffle('x', 0, 'x', 'x'), 
                                mode='low_mem')) 
        input = T.set_subtensor(input[splitPoint:], bn.batch_normalization(input[splitPoint:], 
                                a.dimshuffle('x', 0, 'x', 'x'), b.dimshuffle('x', 0, 'x', 'x'), 
                                mean2.dimshuffle('x', 0, 'x', 'x'), std2.dimshuffle('x', 0, 'x', 'x'),                    
                                mode='low_mem'))                             
    else:    
        input = T.set_subtensor(input[:splitPoint], bn.batch_normalization(input[:splitPoint], a, b, mean1, std1)) 
        input = T.set_subtensor(input[splitPoint:], bn.batch_normalization(input[splitPoint:], a, b, mean2, std2))                    

        
    updateBN = [mean2, var2, normParam['iter']+iterStep]  
    return input, updateBN


def update_bn(mlp, params, updateT1, t1Data, t1Label):
    
    ''' Computation of exact batch normalization parameters for the trained model (referred to test-BN).

    Implemented are three ways to compute the BN parameters: 
        'lazy'      test-BN are approximated by a running average during training
        'default'   test-BN are computed by averaging over activations of params.m samples from training set 
        'proper'    test-BN of k-th layer are computed as in 'default', 
                    however the activations are recomputed by rerunning with test-BN params on all previous layers

    If the setting is 'lazy', this function will not be called, since running average test-BN 
    are computed automatically during training.                

    '''
    
    oldBN, newBN = [{}, {}]
    nSamples1 = t1Data.shape[0]

    batchSizeBN = nSamples1/params.m    
    trainPermBN = range(0, nSamples1)
    np.random.shuffle(trainPermBN)


    # list of layers which utilize BN    
    if params.model == 'convnet':
        allLayers = params.convLayers
        loopOver = filter(lambda i: allLayers[i].bn, range(len(allLayers)))
        print loopOver
    else:
        loopOver = range(params.nLayers-1)
        
    # extract old test-BN parameters, reset new
    oldBN['mean'] = map(lambda i: mlp.h[i].normParam['mean'].get_value(), loopOver)
    oldBN['var'] = map(lambda i: mlp.h[i].normParam['var'].get_value(), loopOver)                    
    newBN['mean'] = map(lambda i: 0.*oldBN['mean'][i], range(len(loopOver)))
    newBN['var'] = map(lambda i: 0.*oldBN['var'][i], range(len(loopOver)))
                     

    # CASE: 'proper' 
    if params.testBN == 'proper':
        
        # loop over layers, loop over examples
        for i in len(range(loopOver)):
            for k in range(0, params.m):
                sampleIndexBN = trainPermBN[(k * batchSizeBN):((k + 1) * (batchSizeBN))]
                _ = updateT1(t1Data[sampleIndexBN], t1Data[0:0], t1Label[sampleIndexBN], t1Label[0:0], 0, 0, 0, 0, 3, 0)                         

                l =loopOver[i]
                newBN['mean'][i] = mlp.h[l].normParam['mean'].get_value() + newBN['mean'][i]
                newBN['var'][i] = mlp.h[l].normParam['var'].get_value() + newBN['var'][i]                   
        np.random.shuffle(trainPermBN)
        biasCorr = batchSizeBN/(batchSizeBN-1)                     
        
        # compute mean, adjust for biases
        newBN['mean'][i] /= params.m
        newBN['var'][i] *= biasCorr/params.m

    # CASE: 'default'                       
    elif params.testBN == 'default': 
        
        # loop over examples
        for k in range(0, params.m):
            sampleIndexBN = trainPermBN[(k * batchSizeBN):((k + 1) * (batchSizeBN))]
            _ = updateT1(t1Data[sampleIndexBN], t1Data[0:0], t1Label[sampleIndexBN], t1Label[0:0], 0, 0, 0, 0, 2, 0)
                        
            newBN['mean'] = map(lambda (i, j): mlp.h[i].normParam['mean'].get_value() + newBN['mean'][j], zip(loopOver, range(len(loopOver))))
            newBN['var'] = map(lambda (i, j): mlp.h[i].normParam['var'].get_value() + newBN['var'][j], zip(loopOver, range(len(loopOver))))                    
                    
        # compute mean, adjust for biases
        biasCorr = batchSizeBN / (batchSizeBN-1)             
        newBN['var'] = map(lambda i: newBN['var'][i]*biasCorr/params.m, range(len(loopOver)))        
        newBN['mean'] = map(lambda i: newBN['mean'][i]/params.m, range(len(loopOver)))

    # updating test-BN parameters, update shared
    map(lambda (i,j): mlp.h[i].normParam['mean'].set_value(newBN['mean'][j]), zip(loopOver, range(len(loopOver))))
    map(lambda (i,j): mlp.h[i].normParam['var'].set_value(newBN['var'][j]), zip(loopOver, range(len(loopOver))))

    # printing an example of previous and updated versions of test-BN
    print 'BN samples: '
    print oldBN['mean'][-1][0], newBN['mean'][-1][0]
    print oldBN['var'][-1][0], newBN['var'][-1][0]
    print oldBN['mean'][1][0], newBN['mean'][1][0]
    print oldBN['var'][1][0], newBN['var'][1][0]

    
    return mlp  
