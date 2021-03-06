import theano.tensor as T
import numpy as np
import theano

from models.layers.mlp_layer import mlp_layer
from training.monitor import stat_monitor

class mlp(object):
    def __init__(self, rng, rstream, x, wantOut, params, useRglrz, bnPhase, globalParams = None):# add cost

        ''' Constructing the mlp model.
        
        Arguments: 
            rng, rstream         - random streams          
            x1, x2       - x batches from T1 and T2 set
            wantOut1, wantOut2   - corresponding labels  
            params               - all model parameters
            graph                - theano variable determining how are BN params computed
            globalParams         - T2 params when one-per-network
       
        '''
                    
        # defining shared variables shared across layers
        if globalParams is None:
            globalParams = {}            

            for rglrz in params.rglrzPerNetwork:                
                tempParam = np.asarray(params.rglrzInitial[rglrz][0], dtype=theano.config.floatX)
                globalParams[rglrz] = theano.shared(value=tempParam, name='%s_%d' % (rglrz, 0), borrow=True)

            for rglrz in params.rglrzPerNetwork1:
                tempParam = np.asarray(params.rglrzInitial[rglrz][0], dtype=theano.config.floatX)
                globalParams[rglrz+str(0)] = theano.shared(value=tempParam, name='%s_%d' % (rglrz, 0), borrow=True)

                tempParam = np.asarray(params.rglrzInitial[rglrz][1], dtype=theano.config.floatX)
                globalParams[rglrz] = theano.shared(value=tempParam, name='%s_%d' % (rglrz, 1), borrow=True)

        # initializations of counters, lists and dictionaries
        h = [] 
        penalty = 0.
        trackT2Params = {}
        for param in params.rglrz: 
            trackT2Params[param] = []
        paramsT1, paramsT2, paramsBN, updateBN = [[],[],[],[]]

        # CONSTRUCT NETWORK
        for i in range(0, params.nLayers):
            
            h.append(mlp_layer(rng=rng, rstream=rstream, index=i, x=x,
                               params=params, globalParams=globalParams, useRglrz=useRglrz, bnPhase=bnPhase))            
            # collect penalty terms
            if 'L2' in params.rglrz:
                tempW = h[i].rglrzParam['L2'] * T.sqr(h[i].W)               
                penalty += T.sum(tempW)
            if 'L1' in params.rglrz:
                tempW = self.rglrzParam['L1'] * T.sum(abs(h[i].W), axis=0)
                penalty += T.sum(tempW)
            if 'LmaxCutoff' in params.rglrz:
                c = self.rglrzParam['LmaxCutoff'] # cutoff
                s = self.rglrzParam['LmaxSlope'] # slope
                tempW = T.sqrt(T.sum(T.sqr(h[i].W), axis=0))
                penalty += T.sum(s*T.sqr(T.maximum(tempW - c, 0)))
                
            # collect T1 params                   
            paramsT1 += h[i].paramsT1
            
            # collect T2 params
            paramsT2 += h[i].paramsT2
            for param in params.rglrz: 
                if (param == 'inputNoise' and i == 0) or (param != 'inputNoise'):
                    trackT2Params[param] += [h[i].rglrzParam[param]]
 
            # cikkect BN params&updates 
            if params.batchNorm and params.activation[i] != 'softmax': 
                paramsBN += h[i].paramsBN
                updateBN += h[i].updateBN

            x = h[-1].output

        # pack variables for output
        for rglrz in globalParams.keys():
            paramsT2 += [globalParams[rglrz]]
        self.paramsT1 = paramsT1
        self.paramsT2 = paramsT2

        self.paramsBN = paramsBN
        self.updateBN = updateBN

        # fix tracking of stats 
        if params.trackStats: 
            self.netStats = stat_monitor(layers = h, params = params)
        else:
            self.netStats = T.constant(0.)           
        self.trackT2Params = trackT2Params

        # output and predicted labels
        self.h = h
        self.y = h[-1].output
        self.guessLabel = T.argmax(self.y, axis=1)
        self.penalty = penalty if penalty != 0. else T.constant(0.)
        self.guessLabel = T.argmax(self.y, axis=1)

        # cost functions
        def stable(x, stabilize=True):
            if stabilize:
                x = T.where(T.isnan(x), 1000., x)
                x = T.where(T.isinf(x), 1000., x)
            return x

        if params.cost == 'categorical_crossentropy':
            def costFun1(y, label):
                return stable(-T.log(y[T.arange(label.shape[0]), label]),
                              stabilize=True)
        else:
            raise NotImplementedError
        if params.cost_T2 in ['categorical_crossentropy', 'sigmoidal', 'hingeLoss']:
            def costFun2(y, label):
                return stable(-T.log(y[T.arange(label.shape[0]), label]),
                              stabilize=True)
        else:
            raise NotImplementedError


        def costFunT1(*args, **kwargs):
            return T.mean(costFun1(*args, **kwargs))
        def costFunT2(*args, **kwargs):
            return T.mean(costFun2(*args, **kwargs))
#        self.y1_avg = self.y1
#        self.guessLabel1_avg = self.guessLabel1

        # cost function
        self.trainCost = costFunT1(self.y, wantOut)
        self.classError = T.mean(T.cast(T.neq(self.guessLabel, wantOut), 'float32'))    
