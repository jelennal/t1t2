import numpy as np
import theano
import theano.tensor as T

from models.layers.conv_layers import conv_layer, pool_layer, average_layer
from models.layers.mlp_layer import mlp_layer
from training.monitor import stat_monitor 
   
zero = theano.shared(value=0., borrow=True)

class convnet(object):
    def __init__(self, rng, rstream, input1, input2, wantOut1, wantOut2, params, graph, globalParams = None):
        
        
        ''' Constructing the convolutional model.
        
        Arguments: 
            rng, rstream         :: random streams          
            input1, input2       :: input batches from T1 and T2 set
            wantOut1, wantOut2   :: corresponding labels  
            params               :: all model parameters
            graph                :: theano variable determining how are BN params computed
            globalParams         :: T2 params when one-per-network
       
        '''
        
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
        i = 0
        h = [] 
        penalty = 0.
        trackT2Params = {}
        for param in params.rglrz: 
            trackT2Params[param] = []
        paramsT1, paramsT2, paramsBN, updateBN = [[],[],[],[]]
        netStats =  {}
        for key in params.activTrack:
            netStats[key] =  []
     
        '''
            Constructing layers.
        '''
        for layer in params.convLayers:            

            # construct layer
            print 'layer ', str(i), ':', layer.type, layer.filter, layer.maps, ' filters'  
            if layer.type == 'conv':
                h.append(conv_layer(rng=rng, rstream=rstream, index=i, splitPoint=splitPoint, input=input,
                                    params=params, globalParams=globalParams, graph=graph,
                                    filterShape=layer.filter, inFilters=layer.maps[0], outFilters=layer.maps[1], stride=layer.stride))                
            elif layer.type == 'pool':
                h.append(pool_layer(rstream=rstream, input=input, params=params, index=i, splitPoint=splitPoint, graph=graph,                      
                                    poolShape=layer.filter, inFilters=layer.maps[0], outFilters=layer.maps[1], stride=layer.stride))
            elif layer.type in ['average', 'average+softmax']:
                h.append(average_layer(rstream=rstream, input=input, params=params, index=i, splitPoint=splitPoint, graph=graph,
                                       poolShape=layer.filter, inFilters=layer.maps[0], outFilters=layer.maps[1], stride=layer.stride))
            elif layer.type == 'softmax':
                h.append(mlp_layer(rng=rng, rstream=rstream, index=i, splitPoint=splitPoint, input=input,
                                   params=params, globalParams=globalParams, graph=graph))

            # collect penalty term
            if layer.type in ['conv', 'softmax'] and ('L2' in params.rglrz):                               
                if 'L2' in params.rglrzPerMap:                
                    tempW = h[-1].rglrzParam['L2'].dimshuffle(0, 'x', 'x', 'x') * T.sqr(h[-1].W)
                else:     
                    tempW = h[-1].rglrzParam['L2'] * T.sqr(h[-1].W)
                penalty += T.sum(tempW) 

            # collect T1 params
            if layer.type in ['conv', 'softmax']:                            
                paramsT1 += h[i].paramsT1
            elif params.batchNorm and params.convLayers[i].bn:
                paramsT1 += h[i].paramsT1

            # collect T2 params
            if params.useT2:
                paramsT2 += h[i].paramsT2
                
            # collect T2 for tracking
            for param in params.rglrz:
                if param == 'inputNoise':
                    if i==0 and layer.noise:
                        trackT2Params[param] += [h[-1].rglrzParam[param]]                        
                    else:
                        trackT2Params[param] += [zero]
                if param == 'addNoise':
                    if layer.noise:
                        trackT2Params[param] += [h[-1].rglrzParam[param]]                        
                    else:
                        trackT2Params[param] += [zero]                    
                if param in ['L1', 'L2', 'Lmax']:    
                    if layer.type in ['conv', 'softmax']:
                        trackT2Params[param] += [h[-1].rglrzParam[param]]                        
                    else:
                        trackT2Params[param] += [zero]                                        
            
            # collect BN params&updates
            if params.batchNorm and params.convLayers[i].bn:
                paramsBN += h[-1].paramsBN
                updateBN += h[-1].updateBN                
            
            input = h[-1].output
            i += 1                   
                                

        # pack variables for output
        for rglrz in globalParams.keys():
            if rglrz in params.rglrzTrain:                
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
        for param in params.rglrz:
            print len(trackT2Params[param]) 
        print '# t1 params: ', len(paramsT1), '# t2 params: ', len(paramsT2) 

        # output and predicted labels
        self.h = h
        self.y = h[-1].output
        self.guessLabel = T.argmax(self.y, axis=1)
        self.penalty = penalty if penalty != 0. else T.constant(0.)

        # split the T1 and T2 batch streams
        self.y1 = self.y[:splitPoint]
        self.y2 = self.y[splitPoint:]
        self.guessLabel1 = T.argmax(self.y1, axis=1)
        self.guessLabel2 = T.argmax(self.y2, axis=1)

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
        self.y1_avg = self.y1
        self.guessLabel1_avg = self.guessLabel1

        # cost function
        self.classError1 = costFunT1(self.y1, wantOut1)
        self.classError2 = costFunT2(self.y2, wantOut2)

