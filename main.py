from time import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import theano
theano.config.exception_verbosity = 'high'
theano.config.optimizer = 'fast_compile'
theano.config.floatX = 'float32'
#from theano.compile.nanguardmode import NanGuardMode

'''     
    Setup the model and train it!

    # Comments:                

'''


from setup import setup
from preprocess.read_preprocess import read_preprocess
from models.mlp import mlp
from models.convnet import convnet
from models.layers.batchnorm import update_bn
from training.schedule import lr_schedule
from training.updates import updates
from training.finite_difference import fd_memory, fd1, fd2, fd3


def run_exp(replace_params={}):

    
    # READ PARAMETERS AND DATA
    params = setup(replace_params)    
    t1Data, t1Label, t2Data, t2Label, vData, vLabel, testD, testL = read_preprocess(params=params)    
    
    # random numbers            
    rng = np.random.RandomState(params.seed)
    rstream = RandomStreams(rng.randint(params.seed+1))

    ''' 
        Construct Theano functions.
        
    '''
    
    # INPUTS   
    graph = T.iscalar('graph')
    phase = T.iscalar('phase')
    if params.model == 'convnet':
        x1 = T.ftensor4('x1')
        x2 = T.ftensor4('x2')
    else:
        x1 = T.matrix('x1')
        x2 = T.matrix('x2')
    trueLabel1 = T.ivector('trueLabel1')
    trueLabel2 = T.ivector('trueLabel2')
    globalLR1 = T.fscalar('globalLR1') 
    globalLR2 = T.fscalar('globalLR2') 
    moment1 = T.fscalar('moment1') 
    moment2 = T.fscalar('moment2') 

    # NETWORK
    if params.model == 'convnet':
        model = convnet(rng=rng, rstream=rstream, input1=x1, input2=x2,
                        wantOut1=trueLabel1, wantOut2=trueLabel2, params=params, graph=graph)
    else:
        model = mlp(rng=rng, rstream=rstream, input1=x1, input2=x2,
                    wantOut1=trueLabel1, wantOut2=trueLabel2, params=params, graph=graph)

    # UPDATES
    updateT1, updateT2, upNormDiff, debugs = updates(mlp=model, params=params,
                                 globalLR1=globalLR1, globalLR2=globalLR2,
                                 momentParam1=moment1, momentParam2=moment2, phase=phase) 
    if params.finiteDiff:                             
        fdm = fd_memory(params=params, model=model)    
        fd_updates1, debugs1 = fd1(mlp=model, fdm=fdm, params=params, globalLR1=globalLR1, globalLR2=globalLR2, 
                                   momentParam1=moment1, momentParam2=moment2) 
        fd_updates2 = fd2(mlp=model, fdm=fdm, params=params, globalLR1=globalLR1, globalLR2=globalLR2, 
                                   momentParam1=moment1, momentParam2=moment2)
        fd_updates3, debugs3 = fd3(mlp=model, fdm=fdm, params=params, globalLR1=globalLR1, globalLR2=globalLR2, 
                                   momentParam1=moment1, momentParam2=moment2)
                                 
    updateBN = []
    if params.batchNorm:
        for param, up in zip(model.paramsBN, model.updateBN):
            updateBN += [(param, up)] 
        
    # THEANO FUNCTIONS 
    updateT1T2 = theano.function(
        inputs = [x1, x2, trueLabel1, trueLabel2, globalLR1, globalLR2, moment1, moment2, graph, phase],
        outputs = [model.classError1, model.guessLabel1, model.classError2, model.guessLabel2] + debugs,
        updates = updateT1 + updateT2 + updateBN,
        on_unused_input='ignore',
#        mode = NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True),
        allow_input_downcast=True)

    updateT1 = theano.function(
        inputs = [x1, x2, trueLabel1, trueLabel2, globalLR1, globalLR2, moment1, moment2, graph, phase],
        outputs = [model.classError1, model.guessLabel1] + debugs,
        updates = updateT1 + updateBN,
        on_unused_input='ignore',
#        mode = NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True),
        allow_input_downcast=True)

    evaluate = theano.function(
        inputs = [x1, x2, trueLabel1, trueLabel2, graph],
        outputs = [model.classError2, model.guessLabel2, model.y2], #+ model.h[-1].debugs,
        on_unused_input='ignore')
# get all these out:
#        outputs = [model.classError1, model.classError2, model.penalty, model.hStat],
        
    # FINITE DIFFERENCE UPDATES   
    if params.finiteDiff:    
        finite_diff1 = theano.function(
            inputs = [x1, x2, trueLabel1, trueLabel2, globalLR1, globalLR2, moment1, moment2, graph],
            outputs = [model.guessLabel1_avg, model.guessLabel2], #+ debugs1,
            updates = fd_updates1 + updateBN,
            on_unused_input='ignore',
    #        mode = NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True),
            allow_input_downcast=True)
        finite_diff2 = theano.function(
            inputs = [x1, x2, trueLabel1, trueLabel2, globalLR1, globalLR2, moment1, moment2, graph],
            updates = fd_updates2,
            on_unused_input='ignore',
    #        mode = NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True),
            allow_input_downcast=True)   
        finite_diff3 = theano.function(
            inputs = [x1, x2, trueLabel1, trueLabel2, globalLR1, globalLR2, moment1, moment2, graph],
    #        outputs = debugs3,
            updates = fd_updates3,
            on_unused_input='ignore',
    #        mode = NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True),
            allow_input_downcast=True)   


    ''' 
        Inializations.
        
    '''

    # INITIALIZE 
    # layers to be read from
    loopOver = range(params.nLayers)
    # initializing training values
    currentT2Batch = 0
    # samples, batches per epoch, etc.
    nSamples1 = t1Data.shape[0]
    nVSamples, nTestSamples  = [vData.shape[0], testD.shape[0]]
    nBatches1  = (nSamples1 / params.batchSize1)
    # permutations
    testPerm = range(0, nTestSamples)
    train1Perm = range(0, nSamples1)
    if params.useT2:
        nSamples2 = t2Data.shape[0]
        train2Perm = range(0, nSamples2)
        nBatches2 = (nSamples2 / params.batchSize2)

    # TRACKING
    # (1) best results
    bestVal = 1.; bestValTst = 1.
    # (2) errors
    tempError1, tempError2, tempCost1, tempCost2 = [[],[], [],[]]
    t1Error, t2Error, validError, testError = [[],[],[],[]]
    t1Cost, t2Cost, penaltyCost, validCost, testCost = [[],[],[],[],[]]
    # (3) activation statistics (per layer)
    trackTemplate = np.empty((0,params.nLayers), dtype = object)
    trackLayers = {}
    for stat in params.activTrack: trackLayers[stat] = trackTemplate
    # (4) penalty, noise, activation parametrization (per layer)
    penalList = ['L1', 'L2', 'LmaxCutoff', 'LmaxSlope', 'LmaxHard']
    noiseList = ['addNoise', 'inputNoise', 'dropOut', 'dropOutB']
    trackPenal = {}
    trackNoise = {}
    trackPenalSTD = {}
    trackNoiseSTD = {}
    trackGrads = {}

    trackGrads['T1'] = []
    trackGrads['T2'] = []
    track1stFeatures = []

    trackRglrzTemplate = np.empty((0,len(loopOver)), dtype = object)
    for param in params.rglrz:
        if param in penalList:
            trackPenal[param] = trackRglrzTemplate
            trackPenalSTD[param] = trackRglrzTemplate
        if param in noiseList:
            trackNoise[param] = trackRglrzTemplate
            trackNoiseSTD[param] = trackRglrzTemplate
    # (5) other
    trackLR1, trackLR2 = [[],[]] # global learning rate for T1 and T2

    params.halfLife = params.halfLife*10000./(params.maxEpoch*nBatches1)
    print 'number of updates total', params.maxEpoch*nBatches1 #maxUpdates
    print 'number of updates within epoch', nBatches1


    ''' 
        Training!!!
        
    '''

    try:
        t_start = time() #
        for i in range(0, params.maxEpoch*nBatches1): # i = nUpdates

            # EPOCHS
            currentEpoch = i / nBatches1
            currentBatch = i % nBatches1 # batch order in the current epoch
            currentProgress = np.around(1.*i/nBatches1, decimals=4)

            # LEARNING RATE SCHEDULES & MOMENTUM
            t = 1.*i/(params.maxEpoch*nBatches1)
            lr1 = np.asarray(params.learnRate1*
                  lr_schedule(fun=params.learnFun1,var=t,halfLife=params.halfLife, start=0),theano.config.floatX)
            lr2 = np.asarray(params.learnRate2*
                  lr_schedule(fun=params.learnFun2,var=t,halfLife=params.halfLife, start=params.triggerT2),theano.config.floatX)
            #if params.triggerT2 > 0: lr2 = min(lr2, lr1)  # alternatively l2*lr1 , for smoother version

            moment1 = np.asarray(params.momentum1[1] - (params.momentum1[1]-(params.momentum1[0]))*
                     lr_schedule(fun=params.momentFun,var=t,halfLife=params.halfLife,start=0), theano.config.floatX)
            moment2 = np.asarray(params.momentum2[1] - (params.momentum2[1]-(params.momentum2[0]))*
                     lr_schedule(fun=params.momentFun,var=t,halfLife=params.halfLife,start=0), theano.config.floatX)

            # PERMUTING T1 AND T2 SETS
            if currentBatch == 0:
                np.random.shuffle(train1Perm)
            if params.useT2 and (currentT2Batch == nBatches2) :
                np.random.shuffle(train2Perm)
                currentT2Batch = 0

            
            # TRAIN T1&T2 -----------------------------------------------------
            if params.useT2:
                # Batches                
                sampleIndex1 = train1Perm[(currentBatch * params.batchSize1):
                                        ((currentBatch + 1) * (params.batchSize1))]
                sampleIndex2 = train2Perm[(currentT2Batch * params.batchSize2):
                                        ((currentT2Batch + 1) * (params.batchSize2))]                
                # finite difference update                        
                if params.finiteDiff:
                    res = finite_diff1(t1Data[sampleIndex1], t2Data[sampleIndex2],
                                   t1Label[sampleIndex1], t2Label[sampleIndex2],
                                   lr1, lr2, moment1, moment2, 0)
                    (y1, y2, debugs) = (res[0], res[1], res[2:])   
                    if  ((i+1) % params.T1perT2) == 0:
                        finite_diff2(t1Data[sampleIndex1], t2Data[sampleIndex2],
                                     t1Label[sampleIndex1], t2Label[sampleIndex2],
                                     lr1, lr2, moment1, moment2, 0)
                        finite_diff3(t1Data[sampleIndex1], t2Data[sampleIndex2],
                                     t1Label[sampleIndex1], t2Label[sampleIndex2],
                                     lr1, lr2, moment1, moment2, 0)                                       
                        tempError2 += [1.*sum(t2Label[sampleIndex2] != y2) / params.batchSize2]
                        currentT2Batch += 1
                # exact update
                else: 
                   doT2 = ((i+1) % params.T1perT2 ==  0) 

                   if  doT2:
                       res = updateT1T2(t1Data[sampleIndex1], t2Data[sampleIndex2],
                                   t1Label[sampleIndex1], t2Label[sampleIndex2],
                                   lr1, lr2, moment1, moment2, 0, 0)
                       (c1, y1, c2, y2, debugs) = (res[0], res[1], res[2], res[3], res[4:])   
                       tempError2 += [1.*sum(t2Label[sampleIndex2] != y2) / params.batchSize2]
                       tempCost2 += [c2]
                       currentT2Batch += 1                       
                   else:
                       res = updateT1(t1Data[sampleIndex1], t2Data[sampleIndex2],
                                   t1Label[sampleIndex1], t2Label[sampleIndex2],
                                   lr1, 0, moment1, 1, 0, 0)
                       (c1, y1, debugs) = (res[0], res[1], res[2:])                          
                tempError1 += [1.*sum(t1Label[sampleIndex1] != y1) / params.batchSize1]                                   
                tempCost1 += [c1]
#               if True in np.isnan(debugs): print 'NANS'
            # TRAIN T1 only ---------------------------------------------------   
            else: 
                sampleIndex1 = train1Perm[(currentBatch * params.batchSize1):
                                          ((currentBatch + 1) * (params.batchSize1))]
                res = updateT1(t1Data[sampleIndex1], t1Data[0:0],
                               t1Label[sampleIndex1], t1Label[0:0],
                               lr1, 0, moment1, 1, 0, 0)
                (c1, y1, debugs) = (res[0], res[1], res[2:])
                tempError1 += [1.*sum(t1Label[sampleIndex1] != y1) / params.batchSize1]
                tempCost1 += [c1]


            #  EVALUATE, PRINT, STORE
            if np.around(currentProgress % (1./params.trackPerEpoch), decimals=4) == 0 \
                    or i == params.maxEpoch*nBatches1 - 1:

                # batchnorm parameters: estimate for the final model
                if (params.batchNorm and (currentEpoch > 1)) \
                   and ((currentEpoch % params.evaluateTestInterval) == 0 or i == (params.maxEpoch*nBatches1 - 1)) \
                   and params.testBN != 'lazy':
                       model = update_bn(model, params, updateT1, t1Data, t1Label)    
                     
#                # EVALUATE: validation set
#                allVar = evaluate(vData[0:0], vData, vLabel[0:0], vLabel, 1)
#                cV, yTest, _ , hStat, _ = allVar[0], allVar[1], allVar[2], allVar[3], allVar[4:]
#                #cV, yTest = allVar[0], allVar[1]
#                tempVError = 1.*sum(yTest != vLabel) / nVSamples
#                tempVError = 7.; cV = 7.
                       
                # EVALUATE: test set - in batches of 1000, ow large to fit onto gpu
                tempError = 0.; tempCost = 0.; nTempSamples = 1000
                if params.useT2 and currentEpoch > 0.9*params.maxEpoch:
                    np.random.shuffle(testPerm)
                    tempIndex = testPerm[:nTempSamples]
                    cT, yTest, _ , _ = evaluate(testD[0:0], testD[tempIndex], testL[0:0], testL[tempIndex], 1)
                    tempError = 1.*sum(yTest != testL[tempIndex]) / nTempSamples
                else:                    
                    for i in range(10):
                        cT, yTest, _ , _ = evaluate(testD[0:0], testD[i*1000:(i+1)*1000], testL[0:0], testL[i*1000:(i+1)*1000], 1)
                        tempError += 1.*sum(yTest != testL[i*1000:(i+1)*1000]) / 1000
                        tempCost += cT
                    tempError /= 10.                     
                    cT = tempCost / 10.
                                                       
                # (2) TRACK: errors                          note: T1 and T2 errors are averaged over training, hence initially can not be compared to valid and test set
                t1Error += [np.mean(tempError1)]; t1Cost += [np.mean(tempCost1)] 
                if params.useT2:
                    t2Error += [np.mean(tempError2)]; t2Cost += [np.mean(tempCost2)]
                testError +=  [tempError]; testCost += [cT]                                        
                #validError += [tempVError]

                # RESET tracked errors
                tempError1 = []
                tempError2 = []
    
                # (4) TRACK: T2 parameter statistics
                if params.useT2:
                    for param in params.rglrz:
                        if param == 'inputNoise':
                            tempParam = map(lambda i: np.mean(model.trackT2Params[param][i].get_value()), range(1))
                            tempParam = np.append(tempParam, np.zeros(len(loopOver)-1))
                        else:
                            tempParam = map(lambda i: np.mean(model.trackT2Params[param][i].get_value()), loopOver)
                        if param in penalList:
                             trackPenal[param] = np.append(trackPenal[param], np.array([tempParam]), axis = 0)
                        elif param in noiseList:
                             trackNoise[param] = np.append(trackNoise[param], np.array([tempParam]), axis = 0)

                # (5) TRACK: global learning rate for T1 and T2
                trackLR1 += [np.log10(lr1)]
                trackLR2 += [np.log10(lr2)]

                # (3) TRACK: unit statistics
#                if params.trackStat:
#                    for key, j in zip(params.activTrack, range(len(params.activTrack))):
#                        trackLayers[key] = np.append(trackLayers[key], np.array([hStat[j]]), axis = 0)
#                   if key == 'wnorm' or key == 'wstd':
#                        print trackLayers[key][-1]                
#                # (6) TRACK: 1st layer features
#                if params.track1stFeatures:
#                    tempW = model.h[0].W.get_value()
#                    track1stFeatures += [tempW[:, :10]]
    
                # PRINT errors and time
                if params.useT2 and ((currentEpoch % params.printInterval) == 0 or 
                                     (i == params.maxEpoch*nBatches1 - 1)):
                    print currentEpoch, ') time=%.f     T1 | T2 | test | penalty ' % ((time() - t_start)/60)
                    
                    print 'ERR    %.3f | %.3f | %.3f | - ' % (
                        t1Error[-1]*100,
                        t2Error[-1]*100,
                        testError[-1]*100)
                    print 'COSTS   %.3f | %.3f | %.3f | ? ' % (
                        t1Cost[-1],
                        t2Cost[-1],
                        testCost[-1])

                    print 'Log[learningRates] ', np.log10(lr1), 'T1 ', np.log10(lr2), 'T2'                        
                    for param in params.rglrzTrain:
                        if param in penalList:
                            print param, trackPenal[param][-1]
                        if param in noiseList:
                            print param, trackNoise[param][-1]
#                    if params.trackStat:        
#                        print 'sparsity: ', trackLayers['spars'][-1]
#                        print '____________________________________________'


                if ((currentEpoch % params.printInterval) == 0 or (i == params.maxEpoch*nBatches1 - 1)):
                    print currentEpoch, 'TRAIN %.2f  TEST %.2f time %.f' % (
                    t1Error[-1]*100, testError[-1]*100, ((time() - t_start)/60))
                    print 'Est. time till end: ', (((time() - t_start)/60) / (currentEpoch+1))*(params.maxEpoch - currentEpoch)

    except KeyboardInterrupt: pass
    time2train = (time() - t_start)/60

    '''
        Prepare variables for output.
    '''
    if params.useT2:

       lastT2 = t2Error[-1]
       allErrors = np.concatenate(([t1Error], [t2Error], [testError]), axis = 0)
       allCosts = np.concatenate(([t1Cost], [t2Cost], [testCost]), axis = 0) # , [penaltyCost]

       outParams = {}       
       for param in params.rglrz:
           if param in penalList:
               outParams[param] = trackPenal[param][-1]
           if param in noiseList:
               outParams[param] = trackNoise[param][-1]
           else: 
               print 'param not tracked, fix!'
    else:

       lastT2 = 0.
       allErrors = np.concatenate(([t1Error], [testError]), axis = 0)
       allCosts = np.concatenate(([t1Cost], [testCost]), axis = 0)
       outParams = {}
       for param in params.rglrz:
           outParams[param] = params.rglrzInitial[param]

    modelName = 'pics/'
    best = min(testError)
    modelName += str(params.nLayers-1)+'x'+str(params.model)+'_best:'+str(best)+'.pdf'

    # saved for plot
    data = { #'setup'  : params, 
          'modelName' : modelName,
               'best' : best,
               'lastEpoch' : (currentEpoch+1),
               'paramsTrained' : params.rglrzTrain,
               'allErrors': allErrors,
               'allCosts': allCosts,
               'trackLayers': trackLayers,
               'trackPenal': trackPenal,
               'trackNoise': trackNoise,
               'trackFeatures': track1stFeatures,
               'trackPenalSTD': trackPenalSTD,
               'trackNoiseSTD': trackNoiseSTD,
               'trackGrads': trackGrads,
               'trackLR1': trackLR1,
               'trackLR2': trackLR2,
               'outParams': outParams,
               }
               
    import pickle;
    file = open(params.saveName,'wb');
    pickle.dump(data, file);
    file.close()

    # prepared for return
    results = {'bestVal': bestVal, # which could be validation or T2
               'bestValTest': best,
               'lastT1': t1Error[-1],
               'lastT2': lastT2,
               'lastVal': None,#validError[-1],
               'lastTest':testError[-1],
               'outParams': outParams,
               'trackGrads': trackGrads,
               'trackPenal': trackPenal,
               'trackNoise': trackNoise,
               'setup'  : params,
               'lastCTest': testCost[-1], 
               'lastCT1': t1Cost[-1],
               'trainTime': time2train,
               }

    return results

if __name__ == '__main__':
    run_exp()