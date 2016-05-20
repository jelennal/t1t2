import theano
import theano.tensor as T


def noise_conditions(params, index, noiseWhere):
    doNoise = (('addNoise' in params.rglrz) or (index == 0 and 'inputNoise' in params.rglrz)) and params.noiseWhere == noiseWhere
    return doNoise           

def dropout_conditions(params, index, noiseWhere):
    doDrop = (('dropOut' in params.rglrz) or ('dropOutB' in params.rglrz))
    return doDrop           

        
def noiseup(input, splitPoint, noiz, noiseType, params, index, rstream):    
    ''' 
        Additive and multiplicative Gaussian Noise    
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


def dropout(input, splitPoint, drop, params, nIn, rstream):
    ''' 
        Dropout noise.     
    '''    
    
    if 'dropOut' in params.rglrz:
        mask = rstream.binomial(n=1, p=(1.-drop), size=input[:splitPoint].shape, dtype=theano.config.floatX)
        input = T.set_subtensor(input[:splitPoint], mask * input[:splitPoint] )
        input = T.set_subtensor(input[splitPoint:], (1.-drop) * input[splitPoint:] )

    elif 'dropOutB' in params.rglrz: # FIX TODO 
        mask = rstream.binomial(n=1, p=(1.-drop), size=input[:splitPoint].shape, dtype=theano.config.floatX)
        input = T.set_subtensor(input[:splitPoint], input[:splitPoint] * mask)
        input = T.set_subtensor(input[splitPoint:], input[splitPoint:] * (1.-drop))    
    return input    


#def noiseup(input, useRglrz, noiz, noiseType, params, index, rstream):    
#    ''' 
#        Additive and multiplicative Gaussian Noise    
#    '''
#
#    if 'inputNoise' in params.rglrzPerMap or 'addNoise' in params.rglrzPerMap:
#        noiz = noiz.dimshuffle('x', 0, 'x', 'x') 
#
#    if noiseType == 'multi1':                
#        input = input*(1. + noiz*rstream.normal(input.shape, dtype=theano.config.floatX))
#    elif noiseType == 'multi0':                
#        input = input*(noiz*rstream.normal(input.shape, dtype=theano.config.floatX))
#    else:
#        input = input + noiz*rstream.normal(input.shape, dtype=theano.config.floatX)
#    return input
#
#def dropout(input, useRglrz, drop, params, rglrzParam, nIn, rstream):
#    ''' 
#        Dropout noise.     
#    '''    
#    doDrop = useRglrz
#    scaleLayer = 1 - useRglrz
#
#    if 'dropOut' in params.rglrz:
#        mask = rstream.binomial(n=1, p=(1.-drop), size=input.shape, dtype=theano.config.floatX)
#        input = input * (1*(1-doDrop)+mask*doDrop)
#        input = input * (1-drop*scaleLayer)
#
#    elif 'dropOutB' in params.rglrz:
#        mask = rstream.binomial(n=1, p=(1.-drop), size=(nIn,), dtype=theano.config.floatX)
#        input = input * (1*(1-doDrop)+mask*doDrop)
#        input = input * (1-drop*scaleLayer)
#    return input


