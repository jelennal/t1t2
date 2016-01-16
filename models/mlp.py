import theano.tensor as T
import numpy as np
import theano
from models.layers.mlp_layer import mlp_layer
import utils
#import numpy as np
#import theano
#from theano.tensor.shared_randomstreams import RandomStreams

class mlp(object):
    def __init__(self, rng, rstream, input1, input2, wantOut1, wantOut2, params, graph, globalParams = None): # add cost

        ''' Constructing the mlp model.
        
        Arguments: 
            rng, rstream         - random streams          
            input1, input2       - input batches from T1 and T2 set
            wantOut1, wantOut2   - corresponding labels  
            params               - all model parameters
            graph                - theano variable determining how are BN params computed
            globalParams         - T2 params when one-per-network
       
        '''
#        # Multiply the input if M > 0
#        if params.MM > 1:
#            input1 = T.extra_ops.repeat(input1, params.MM, axis=0)
#            wantOut1 = T.extra_ops.repeat(wantOut1, params.MM, axis=0)
    
                # concatenating input streams from T1 and T2 batches 
        splitPoint = wantOut1.shape[0]
        input = T.concatenate([input1, input2], axis=0)
                    
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
        netStats =  {}
        for key in params.activTrack:
            netStats[key] =  []


        # CONSTRUCT NETWORK
        for i in range(0, params.nLayers):
            
            h.append(mlp_layer(rng=rng, rstream=rstream, index=i,
                                 splitPoint=splitPoint, input=input,
                                 paramsL=params, globalParams=globalParams, graph=graph))

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

            input = h[-1].output


        # pack variables for output
        for rglrz in globalParams.keys():
            paramsT2 += [globalParams[rglrz]]
        self.paramsT1 = paramsT1
        self.paramsT2 = paramsT2

        self.paramsBN = paramsBN
        self.updateBN = updateBN

        # fix tracking of stats 
        self.hStat = input
        self.trackT2Params = trackT2Params
        print len(trackT2Params[param]) 


        # output and predicted labels
        self.h = h
        self.y = h[-1].output
        self.guessLabel = T.argmax(self.y, axis=1)
        self.penalty = penalty if penalty != 0. else T.constant(0.)
#        self.penaltyMaxParams = reduce(lambda x, y: x.update(y) or x,
#                                       [l.penaltyMaxParams for l in h], {})

        # split the T1 and T2 batch streams
        self.y1 = self.y[:splitPoint]
        self.y2 = self.y[splitPoint:]
        self.guessLabel1 = T.argmax(self.y1, axis=1)
        self.guessLabel2 = T.argmax(self.y2, axis=1)

        # cost functions
        def stable(inp, stabilize=True):
            if stabilize:
                inp = T.where(T.isnan(inp), 1000., inp)
                inp = T.where(T.isinf(inp), 1000., inp)
            return inp

        if params.cost == 'categorical_crossentropy':
            def costFun1(y, label):
                return stable(-T.log(y[T.arange(label.shape[0]), label]),
                              stabilize=False)
                # return stable(T.nnet.categorical_crossentropy(y, label),
                #               stabilize=False)
        else:
            raise NotImplementedError
        if params.cost_T2 in ['crossEntropy', 'sigmoidal', 'hingeLoss']:
            def costFun2(y, label):
                return stable(-T.log(y[T.arange(label.shape[0]), label]),
                              stabilize=False)
                # return stable(T.nnet.categorical_crossentropy(y, label),
                #               stabilize=False)
        else:
            raise NotImplementedError


        if params.MM == 1:
            def costFunT1(*args, **kwargs):
                return T.mean(costFun1(*args, **kwargs))
            def costFunT2(*args, **kwargs):
                return T.mean(costFun2(*args, **kwargs))
            self.y1_avg = self.y1
            self.guessLabel1_avg = self.guessLabel1
# ------------------------------------------------------------ cases: M>1
#        else:
#            def costFunT1(*args, **kwargs):
#                raw_cost = costFun1(*args, **kwargs)
#                print "Cost dimensionality: %d with duplicates: %d" % (raw_cost.ndim,
#                                                                       params.MM)
#                if raw_cost.ndim == 1:
#                    raw_cost = raw_cost.reshape((raw_cost.shape[0] // params.MM,
#                                                 params.MM))
#                elif raw_cost.ndim == 2:
#                    raw_cost = raw_cost.reshape((raw_cost.shape[0] // params.MM,
#                                                 params.MM, raw_cost.shape[1]))
#                    raw_cost = T.sum(raw_cost, axis=2)
#                raw_cost = -T.mean(utils.LogMeanExp(-raw_cost, axis=1))
#                return raw_cost
#            def costFunT2(*args, **kwargs):
#                return T.mean(costFun1(*args, **kwargs))
#
#            # Also take the mean of the guesses
#            shap = (self.y1.shape[0] // params.MM, params.MM)
#            self.y1_avg = self.y1.reshape(tuple([self.y1.shape[0] // params.MM,
#                                           params.MM] + [self.y1.shape[s] for s in xrange(1, self.y1.ndim)]),
#                                          ndim=self.y1.ndim + 1)
#            self.y1_avg = T.mean(self.y1_avg, axis=1)
#            self.guessLabel1_avg = T.argmax(self.y1_avg, axis=1)

        # cost function
        self.classError1 = costFunT1(self.y1, wantOut1)
        self.classError2 = costFunT2(self.y2, wantOut2)
        
        

# TO FIX, add in separate function        
        
        
#                    # tracking network statistics
#            tempMean = T.mean(input, axis = 0)
#            tempSTD = T.std(input, axis = 0)
#            tempMax = T.max(input, axis = 0)
#            
#            if i == 0:
#                tempSpars = T.mean(T.le(input, -0.2)) # different for input
#                tempRNoise = T.switch(T.le(tempSTD, eps), eps, 1. / tempSTD)                      
#            else:
#                tempSpars = T.mean(T.le(input, eps))
#                tempRNoise = T.switch(T.le(tempSTD, eps), eps, 1. / tempSTD)                      
#            if (key == 'rnoise' or key == 'rnstd') and 'addNoise' in params.rglrz:
#                 tempRNoise *= h[i].rglrzParam['addNoise']
#            else:
#                 tempRNoise *= 0       
#
#            for key in params.activTrack:
#                   statistic = {'mean': T.mean(tempMean),
#                                 'std': T.mean(tempSTD),
#                                 'max': T.max(tempMax),
#                               'const': T.mean(T.le(tempSTD, eps)), 
#                               'spars': tempSpars,
#                               'wmean': T.mean(abs(h[i].W)),
#                                'wstd': T.std(h[i].W),
#                                'wmax': T.max(abs(h[i].W)),
#                              'rnoise': T.mean(tempRNoise),
#                               'rnstd': T.std(tempRNoise),
#                               'bias' : T.mean(h[i].b),
#                                   'a': T.mean(h[i].a),
##                               'bstd' : T.std(h[i].b),
##                                'astd': T.std(h[i].a),
#
#                               }[key]
#  
#                   netStats[key] +=  [statistic]
#                   

#        allStats = [] 
#        if params.trackStat:
#            for key in params.activTrack: allStats += [netStats[key]]
#            self.hStat = T.stacklists(allStats)
#        else: 
#            self.hStat =  h[-1].output
#
        
        