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

from setup import setup
from preprocess.read_preprocess import read_preprocess
from models.mlp import mlp
from models.convnet import convnet
from models.layers.batchnorm import update_bn
from training.schedule import lr_schedule
from training.updates import updates
from training.finite_difference import fd_memory, fd1, fd2, fd3


def run_exp(replace_params={}):

    params = setup(replace_params)    
    t1Data, t1Label, t2Data, t2Label, vData, vLabel, testD, testL = read_preprocess(params=params)    
        
    print params.rglrzLR
    print '- learning rates, style:'
    print params.learnRate1, params.learnRate2, params.learnFun1, params.learnFun2, params.opt1, params.opt2
    print '- initial T2 params:'
    for key in params.rglrz:
        print params.rglrzInitial[key]
    print params.noiseT1
        


    # INITIALIZE MODEL------------------------------------------------------
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
    globalLR1 = T.fscalar('globalLR1') # current training period
    globalLR2 = T.fscalar('globalLR2') # current training period

    moment1 = T.fscalar('moment1') # current training period
    moment2 = T.fscalar('moment2') # current training period

    rng = np.random.RandomState(params.seed)
    rstream = RandomStreams(rng.randint(params.seed+1))
    
    if params.model == 'convnet':
        model = convnet(rng=rng, rstream=rstream, input1=x1, input2=x2,
                      wantOut1=trueLabel1, wantOut2=trueLabel2, params=params, graph=graph)
    else:
        model = mlp(rng=rng, rstream=rstream, input1=x1, input2=x2,
                          wantOut1=trueLabel1, wantOut2=trueLabel2, params=params, graph=graph)

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
        
    # THEANO FUNCTIONS -----------------------------------------------------
    updateT1T2 = theano.function(
        inputs = [x1, x2, trueLabel1, trueLabel2, globalLR1, globalLR2, moment1, moment2, graph, phase],
        outputs = [model.guessLabel1, model.guessLabel2] + debugs,
        updates = updateT1 + updateT2 + updateBN,
        on_unused_input='ignore',
#        mode = NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True),
        allow_input_downcast=True)

    updateT1 = theano.function(
        inputs = [x1, x2, trueLabel1, trueLabel2, globalLR1, globalLR2, moment1, moment2, graph, phase],
        outputs = [model.guessLabel1] + debugs,
        updates = updateT1 + updateBN,
        on_unused_input='ignore',
#        mode = NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True),
        allow_input_downcast=True)


    evaluate = theano.function(
        inputs = [x1, x2, trueLabel1, trueLabel2, graph],
        outputs = [model.classError2, model.guessLabel2, model.y2, model.hStat], #+ model.h[-1].debugs,
        on_unused_input='ignore')

    evaluateT1T2 = theano.function(
        inputs = [x1, x2, trueLabel1, trueLabel2, graph],
        outputs = [model.classError1, model.classError2, model.penalty, model.hStat],
        on_unused_input='ignore',
        allow_input_downcast=True)
        
        
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


    # INITIALIZE 
    # layers to be read from
    if params.model == 'convnet':
        loopOver = filter(lambda i: params.convLayers[i].type =='conv', range(len(params.convLayers)))
        loopOver = range(len(loopOver))
    else:
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
    bestVal = 1.
    bestValTst = 1.
    # (2) errors
    tempError1, tempError2 = [[],[]]
    train1Error, train2Error, validError, testError = [[],[],[],[]]
    train1Cost, train2Cost, penaltyCost, validCost, testCost = [[],[],[],[],[]]
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

    # --------------------------------------------------------------- TRAINING
    try:
        t_start = time() # looping over all the batches in the training set;
        for i in range(0, params.maxEpoch*nBatches1): # i = nUpdates

            # --------------------------------------------- --------------------- ------  epochs, progress
            currentEpoch = i / nBatches1
            currentBatch = i % nBatches1 # batch order in the current epoch
            currentProgress = np.around(1.*i/nBatches1, decimals=4)
            t = 1.*i/(params.maxEpoch*nBatches1)

            # -------------------------------------------- ----------------------- ----- outside learning rates
            lr1 = np.asarray(params.learnRate1*
                  lr_schedule(fun=params.learnFun1,var=t,halfLife=params.halfLife, start=0),theano.config.floatX)
            lr2 = np.asarray(params.learnRate2*
                  lr_schedule(fun=params.learnFun2,var=t,halfLife=params.halfLife, start=params.triggerT2),theano.config.floatX)
            #if params.triggerT2 > 0: lr2 = min(lr2, lr1)  # alternatively l2*lr1 , for smoother version

            moment1 = np.asarray(params.momentum1[1] - (params.momentum1[1]-(params.momentum1[0]))*
                     lr_schedule(fun=params.momentFun,var=t,halfLife=params.halfLife,start=0), theano.config.floatX)
            moment2 = np.asarray(params.momentum2[1] - (params.momentum2[1]-(params.momentum2[0]))*
                     lr_schedule(fun=params.momentFun,var=t,halfLife=params.halfLife,start=0), theano.config.floatX)

            # ------------------------------------------------------------------------- permute T1, permute T2
            if currentBatch == 0:
                np.random.shuffle(train1Perm)
            if params.useT2 and (currentT2Batch == nBatches2) :
                np.random.shuffle(train2Perm)
                currentT2Batch = 0


            if params.useT2:
                # train T1 & T2 ------------------------------------------------------- USES T2
               sampleIndex1 = train1Perm[(currentBatch * params.batchSize1):
                                        ((currentBatch + 1) * (params.batchSize1))]
               sampleIndex2 = train2Perm[(currentT2Batch * params.batchSize2):
                                        ((currentT2Batch + 1) * (params.batchSize2))]
                                    
               if params.finiteDiff: # ------------------------------------------------ T2 + FD
                   
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

               else: # ---------------------------------------------------------------- T2 + EXACT
                   doT2 = ((i+1) % params.T1perT2 ==  0) 

                   if  doT2:
                       res = updateT1T2(t1Data[sampleIndex1], t2Data[sampleIndex2],
                                   t1Label[sampleIndex1], t2Label[sampleIndex2],
                                   lr1, lr2, moment1, moment2, 0, 0)

                       (y1, y2, debugs) = (res[0], res[1], res[2:])   

                       tempError2 += [1.*sum(t2Label[sampleIndex2] != y2) / params.batchSize2]
                       currentT2Batch += 1
                       
                   else:
                       res = updateT1(t1Data[sampleIndex1], t2Data[sampleIndex2],
                                   t1Label[sampleIndex1], t2Label[sampleIndex2],
                                   lr1, 0, moment1, 1, 0, 0)
                       (y1, debugs) = (res[0], res[1:])   
                       
               tempError1 += [1.*sum(t1Label[sampleIndex1] != y1) / params.batchSize1]                                   
#               if True in np.isnan(debugs): print 'NANS'

            else: # ------------------------------------------------------------------- NO T2

               # train T1
               sampleIndex1 = train1Perm[(currentBatch * params.batchSize1):
                                    ((currentBatch + 1) * (params.batchSize1))]

               res = updateT1(t1Data[sampleIndex1], t1Data[0:0],
                                   t1Label[sampleIndex1], t1Label[0:0],
                                   lr1, 0, moment1, 1, 0, 0)
               (y1, debugs) = (res[0], res[1:])

               tempError1 += [1.*sum(t1Label[sampleIndex1] != y1) / params.batchSize1]


            # ---------------------------------- evaluate test, save results, show results
            if np.around(currentProgress % (1./params.trackPerEpoch), decimals=4) == 0 \
                    or i == params.maxEpoch*nBatches1 - 1:

                # BATCHNORM: estimate for test over all the train data
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
                 # most of the time, evaluate just a random sample from 10,000                    
                tempError = 0.; nTempSamples = 1000
                if params.useT2 and currentEpoch > 0.9*params.maxEpoch:
                    np.random.shuffle(testPerm)
                    tempIndex = testPerm[:nTempSamples]
                    cT, yTest, _ , _ = evaluate(testD[0:0], testD[tempIndex], testL[0:0], testL[tempIndex], 1)
                    tempError = 1.*sum(yTest != testL[tempIndex]) / nTempSamples
                else:                    
                    for i in range(10):
                        cT, yTest, _ , _ = evaluate(testD[0:0], testD[i*1000:(i+1)*1000], testL[0:0], testL[i*1000:(i+1)*1000], 1)
                        tempError += 1.*sum(yTest != testL[i*1000:(i+1)*1000]) / 1000
                    tempError /= 10.                     

                
#                # (6) TRACK: gradients
#                if params.batchNorm and not params.aFix:
#                    nTrained = 3*(params.nLayers) - 1 
#                else:
#                    nTrained = 2*(params.nLayers)                 
#                
#                if params.trackGrads:                    
#                    if trackGrads['T1'] == []:
#                        if not params.useT2:
#                            trackGrads['T1'] = np.log10([debugs])
#                            print 'gradworks!', debugs[0]
#                        else:
#                            trackGrads['T1'] = [debugs[:nTrained]]
#                            trackGrads['T2'] = [debugs[nTrained:]]
#
#                            
#                    else:
#                        if not params.useT2:
#                            trackGrads['T1'] = np.append(trackGrads['T1'], np.log10([debugs]), axis = 0)
#                            if params.showGrads:
#                                print np.around([debugs], 3)
#                                print currentEpoch, ') time=%.f  |  T1 %.2f | test %.2f  ' % ((time() - t_start)/60, train1Error[-1]*100, tempError*100)
#                        else:
#                            trackGrads['T1'] = np.append(trackGrads['T1'], [debugs[:nTrained]], axis = 0)                    
#                            trackGrads['T2'] = np.append(trackGrads['T2'], [debugs[nTrained:]], axis = 0)
#                            if params.showGrads:
#                                print np.around([debugs[:nTrained]], 3)
#                                print np.around([debugs[nTrained:]], 3)
                                  
                                                       
                # (2) TRACK: errors                          note: T1 and T2 errors are averaged over training, hence initially can not be compared to valid and test set
                train1Error += [np.mean(tempError1)]
                if params.useT2:
                    train2Error += [np.mean(tempError2)]
                testError +=  [tempError]
                                        
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
#                    for param, j in zip(params.rglrzTrain, range(len(params.rglrzTrain))):
#                        paramsT2 = map(lambda i: np.mean(model.paramsT2[i*len(params.rglrzTrain)+j].get_value()), range(params.nLayers))
#                        paramsT2STD = map(lambda i: np.std(model.paramsT2[i*len(params.rglrzTrain)+j].get_value()), range(params.nLayers))
#                        if param in penalList:
#                             trackPenal[param] = np.append(trackPenal[param], np.array([paramsT2]), axis = 0)
#                             trackPenalSTD[param] = np.append(trackPenalSTD[param], np.array([paramsT2STD]), axis = 0)
#                        elif param in noiseList:
#                             trackNoise[param] = np.append(trackNoise[param], np.array([paramsT2]), axis = 0)
#                             trackNoiseSTD[param] = np.append(trackNoiseSTD[param], np.array([paramsT2STD]), axis = 0)

                # (1) TRACK: best performance
#                if params.useT2:
#                    if train2Error[len(train2Error)-1] < bestVal:
#                       bestVal, bestValTst = [train2Error[len(train2Error)-1], tempError]
#                else:
#                    if tempVError < bestVal:
#                       bestVal, bestValTst = [tempVError, tempError]

                # (5) TRACK: global learning rate for T1 and T2
                trackLR1 += [np.log10(lr1)]
                trackLR2 += [np.log10(lr2)]

                # (3) TRACK: unit statistics
#                if params.trackStat:
#                    for key, j in zip(params.activTrack, range(len(params.activTrack))):
#                        trackLayers[key] = np.append(trackLayers[key], np.array([hStat[j]]), axis = 0)
#                   if key == 'wnorm' or key == 'wstd':
#                        print trackLayers[key][-1]
                
                # (6) TRACK: 1st layer features
                if params.track1stFeatures:
                    tempW = model.h[0].W.get_value()
                    track1stFeatures += [tempW[:, :10]]
    
    
                # PRINT errors and time
                if params.useT2 and ((currentEpoch % params.printInterval) == 0 or 
                                     (i == params.maxEpoch*nBatches1 - 1)):
                    print currentEpoch, ') time=%.f     T1 | T2 | test | penalty ' % ((time() - t_start)/60)
                    
                    print 'ERR    %.3f | %.3f | %.3f | - ' % (
                        train1Error[-1]*100,
                        train2Error[-1]*100,
                        testError[-1]*100)
#                    print 'COSTS   %.3f | %.3f | %.3f | %.3f' % (
#                         c1, c2, cT, p)

                    print 'Log[learningRates] ', np.log10(lr1), 'T1 ', np.log10(lr2), 'T2'                        
                    for param in params.rglrzTrain:
                        if param in penalList:
                            print param, trackPenal[param][-1]
                        if param in noiseList:
                            print param, trackNoise[param][-1]
#                    if params.trackStat:        
#                        print 'sparsity: ', trackLayers['spars'][-1]
#                        print '____________________________________________'


                # SHOW: best results -----------------------------------------------
    #            if (i % params.printBest) == 0 or (i == params.maxEpoch*nBatches1 - 1):
                if ((currentEpoch % params.printInterval) == 0 or (i == params.maxEpoch*nBatches1 - 1)):
                    print currentEpoch, 'TRAIN %.2f  TEST %.2f time %.f' % (
                    train1Error[-1]*100, testError[-1]*100, ((time() - t_start)/60))
                    print 'Est. time till end: ', (((time() - t_start)/60) / (currentEpoch+1))*(params.maxEpoch - currentEpoch)

    except KeyboardInterrupt: pass
    time2train = (time() - t_start)/60


    # prepare for output  -----------------------------------------------------


    best = bestValTst
    if params.useT2:
       lastT2 = train2Error[-1]
       allErrors = np.concatenate(([train1Error], [train2Error], [testError]), axis = 0)
       allCosts = np.concatenate(([train1Cost], [train2Cost], [testCost], [penaltyCost]), axis = 0)
       outParams = {}
       
       for param in params.rglrz:
             if param == 'inputNoise':
                 tempParam = map(lambda i: model.trackT2Params[param][i].get_value(), range(1))
                 tempParam = np.append(tempParam, np.zeros(params.nLayers-1))
             else:
#                 tempParam = map(lambda i: np.mean(model.trackT2Params[param][i].get_value()), range(params.nLayers))
#                 tempParam[0] = model.trackT2Params[param][0].get_value()                                  
                 tempParam = map(lambda i: model.trackT2Params[param][i].get_value(), loopOver)
                 
             outParams[param] = tempParam
             if params.model == 'convnet':
                j = 0; temp = []
                for i in range(params.nLayers):
                    if i in loopOver:
                        temp += [tempParam[j]]
                        j += 1
                    else:
                        temp += [0.]
                outParams[param] = temp
                print temp
#       for param in params.rglrzTrain:
#           if param in penalList: outParams[param] = trackPenal[param][-1]
#           if param in noiseList: outParams[param] = trackNoise[param][-1]

    else:
       lastT2 = 0.
       allErrors = np.concatenate(([train1Error], [testError]), axis = 0)
       allCosts = np.concatenate(([train1Cost], [testCost]), axis = 0)
       outParams = {}
       for param in params.rglrz:
           outParams[param] = params.rglrzInitial[param]


    modelName = 'pics/'
    modelName += str(params.nLayers-1)+'x'+str(params.model)+'_best:'+str(best)+'.pdf'
    # import pdb; pdb.set_trace()


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
               'lastT1': train1Error[-1],
               'lastT2': lastT2,
               'lastVal': None,#validError[-1],
               'lastTest':testError[-1],
               'outParams': outParams,
               'trackGrads': trackGrads,
               'trackPenal': trackPenal,
               'trackNoise': trackNoise,
               'setup'  : params,
#               'lastCTest': testCost[-1], 
#               'lastCT1': train1Cost[-1],
               'trainTime': time2train,
               }

    return results


# ======================================================= will we use it?
#def search():
#    replace_params = {
#        'train_T2_gradient_jacobian': True,
#        'useT2': 1,
##        'rglrzTrain': ['L2']
#    }
#
#
#    # TODO: implement loop with the different hyperparameter values
#    n_iters = 100
#
#    res = run_exp(do_plot=True, replace_params=replace_params)
#    raise NotImplemented

if __name__ == '__main__':
    run_exp()


                # EVALUATE: T1 and T2 set                              only a sample from T1, cause sloooooooooooow
#                if params.useT2:
#                    tempSample1 = train1Perm[-1000:]
#                    tempSample2 = train2Perm[-1000:]
#                    [c1, c2, p, hStat] = evaluateT1T2(t1Data[tempSample1], t2Data[tempSample2],
#                                                      t1Label[tempSample1], t2Label[tempSample2], 0)
#                    # (2) TRACK: costs
#                    train1Cost += [c1]
#                    train2Cost += [c2]
#                    validCost += [cV]
#                    testCost += [cT]
#                    penaltyCost += [p]
#                    # angleCost += [updiff]
#                    # TRACK: error for T2
#                    train2Error += [np.mean(tempError2)]#                    
#                else:
#                    tempSample1 = train1Perm[-1000:]
#                    [c1, _, p, hStat] = evaluateT1T2(t1Data[tempSample1], t1Data[0:0],
#                                                      t1Label[tempSample1], t1Label[0:0], 0)
#                    testCost += [cT]
#                    train1Cost += [c1]
