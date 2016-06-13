from time import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano.sandbox.rng_mrg
import theano

theano.config.exception_verbosity = 'high'
theano.config.floatX = 'float32'
#theano.warn_foat64 = 'warn'
#theano.config.optimizer = 'fast_compile' # ALT: 'fast_run'import the
#theano.config.optimizer_including=cudnn

# run with: CUDA_LAUNCH_BLOCKING=1 python
#theano.config.profile = True
#from theano.compile.nanguardmode import NanGuardMode  # mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True)

def detect_nan(i, node, fn):
    for output in fn.outputs:
        if (not isinstance(output[0], np.random.RandomState) and 
           not (hasattr(node, 'op') or isinstance(node.op, (theano.sandbox.rng_mrg.GPU_mrg_uniform, theano.sandbox.cuda.basic_ops.GpuAllocEmpty)))):
            try:
                has_nans = np.isnan(output[0]).any() or np.isinf(output[0]).any()
            except TypeError:
                has_nans = False
            if not has_nans:
                continue           
            print('*** NaN detected ***')
            theano.printing.debugprint(node, depth=3)
            print(type(node), node.op, type(node.op))
            print('Inputs : %s' % [input[0] for input in fn.inputs])
            print'Input shape',  [input[0].shape for input in fn.inputs]
            print('Outputs: %s' % [output[0] for output in fn.outputs])
            print'Output shape',  [output[0].shape for output in fn.outputs]
            print 'NaN # :', [np.sum(np.isnan(output[0])) for output in fn.outputs]  
            print 'Inf # :', [np.sum(np.isinf(output[0])) for output in fn.outputs]  
            print 'NaN location: ', np.argwhere(np.isnan(output[0])), ', Inf location: ', np.argwhere(np.isinf(output[0]))            
            import pdb; pdb.set_trace()
            raise ValueError


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
from training.monitor import t2_extract, grad_extract, stat_extract


def run_exp(replace_params={}):
    
    # READ PARAMETERS AND DATA
    params = setup(replace_params)    
    t1Data, t1Label, t2Data, t2Label, vData, vLabel, testD, testL = read_preprocess(params=params)
    np.savez('preprocessed_cifar.npz',
             X_train=t1Data,
             Y_train=t1Label,
             X_t2=t2Data,
             Y_t2=t2Label,
             X_v=vData,
             Y_v=vLabel,
             X_test=testD,
             Y_test=testL)
    return
    
    # random numbers            
    rng = np.random.RandomState(params.seed)
    rstream = RandomStreams(rng.randint(params.seed+1)+1)

    ''' 
        Construct Theano functions        

    '''    
    # INPUTS       
    useRglrz = T.fscalar('useRglrz')
    bnPhase = T.fscalar('bnPhase')
    if params.model == 'convnet':
        x = T.ftensor4('x')
    else:
        x = T.matrix('x')
    trueLabel = T.ivector('trueLabel')
    globalLR1 = T.fscalar('globalLR1') 
    globalLR2 = T.fscalar('globalLR2') 
    moment1 = T.fscalar('moment1') 
    moment2 = T.fscalar('moment2')                         

    # NETWORK
    if params.model == 'convnet':
        model = convnet(rng=rng, rstream=rstream, x=x, wantOut=trueLabel, 
                        params=params, useRglrz=useRglrz, bnPhase=bnPhase)
    else:
        model = mlp(rng=rng, rstream=rstream, x=x, wantOut=trueLabel,
                    params=params, useRglrz=useRglrz, bnPhase=bnPhase)

    # UPDATES
    updateT1, updateT2, updateC2grad, grads = updates(mlp=model, params=params,
                                                      globalLR1=globalLR1, globalLR2=globalLR2,
                                                      momentParam1=moment1, momentParam2=moment2)                                  
    updateBN = []
    if params.batchNorm:
        for param, up in zip(model.paramsBN, model.updateBN):
            updateBN += [(param, up)] 
                        

    updateT1 = theano.function(
        inputs = [x, trueLabel, globalLR1, moment1, useRglrz, bnPhase],
        outputs = [model.trainCost, model.guessLabel] + grads,
        updates = updateT1 + updateBN,
#        mode=theano.compile.MonitorMode(post_func=detect_nan),                
        on_unused_input='ignore',
        allow_input_downcast=True)

    updateT2part1 = theano.function(
        inputs = [x, trueLabel, globalLR1, moment1, useRglrz, bnPhase],
        outputs = [model.trainCost, model.guessLabel] + grads,
        updates = updateC2grad,
#        mode=theano.compile.MonitorMode(post_func=detect_nan),                
        on_unused_input='ignore',
        allow_input_downcast=True)

    updateT2part2 = theano.function(
        inputs = [x, trueLabel, globalLR1, moment1, globalLR2, moment2, useRglrz, bnPhase],
        outputs = [model.trainCost, model.guessLabel] + grads,
        updates = updateT2,
#        mode=theano.compile.MonitorMode(post_func=detect_nan),                
        on_unused_input='ignore',
        allow_input_downcast=True)

    evaluate = theano.function(
        inputs = [x, trueLabel, useRglrz, bnPhase],
        outputs = [model.trainCost, model.guessLabel, model.penalty, model.netStats],
        on_unused_input='ignore',
        allow_input_downcast=True)        

    evaluateBN = theano.function(
        inputs = [x, useRglrz, bnPhase],
        updates = updateBN,
        on_unused_input='ignore',
#        mode=theano.compile.MonitorMode(post_func=detect_nan),                
        allow_input_downcast=True)        

               
    ''' 
        Inializations
        
    '''

    # INITIALIZE 
    # layers to be read from
    loopOver = range(params.nLayers)
    # initializing training values
    currentT2Batch = 0
    # samples, batches per epoch, etc.
    nSamples1 = t1Data.shape[0]
    nVSamples, nTestSamples  = [vData.shape[0], testD.shape[0]]
    nBatches1  = nSamples1 / params.batchSize1
    # permutations
    testPerm = range(0, nTestSamples)
    train1Perm = range(0, nSamples1)
    if params.useT2:
        nSamples2 = t2Data.shape[0]
        train2Perm = range(0, nSamples2)
        nBatches2 = nSamples2 / params.batchSize2

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
    penalList = ['L1', 'L2', 'Lmax', 'LmaxCutoff', 'LmaxSlope', 'LmaxHard']
    noiseList = ['addNoise', 'inputNoise', 'dropOut', 'dropOutB']
    sharedNames = [p.name for p in model.paramsT1] + [p.name for p in model.paramsT2] 
    print sharedNames

    trackPenal = {}; trackPenalSTD = {} 
    trackNoise = {}; trackNoiseSTD = {}            
    trackGrads = {}
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
    trackLR1, trackLR2 = [[],[]] 

    params.halfLife = params.halfLife*10000./(params.maxEpoch*nBatches1)
    print 'number of updates total', params.maxEpoch*nBatches1 
    print 'number of updates within epoch', nBatches1

    

    ''' 
        Training!!!
        
    '''
    lastUpdate = params.maxEpoch*nBatches1 - 1

    try:
        t_start = time() #
        for i in range(0, params.maxEpoch*nBatches1): # i = nUpdates

            # EPOCHS
            currentEpoch = i / nBatches1
            currentBatch = i % nBatches1 # batch order in the current epoch
            currentProgress = np.around(1.*i/nBatches1, decimals=4)

            '''
                Learning rate and momentum schedules.
            '''
            
            t = 1.*i/(params.maxEpoch*nBatches1)
            lr1 = np.asarray(params.learnRate1*
                  lr_schedule(fun=params.learnFun1,var=t,halfLife=params.halfLife, start=0),theano.config.floatX)
            lr2 = np.asarray(params.learnRate2*
                  lr_schedule(fun=params.learnFun2,var=t,halfLife=params.halfLife, start=params.triggerT2),theano.config.floatX)

            moment1 = np.asarray(params.momentum1[1] - (params.momentum1[1]-(params.momentum1[0]))*
                     lr_schedule(fun=params.momentFun,var=t,halfLife=params.halfLife,start=0), theano.config.floatX)
            moment2 = np.asarray(params.momentum2[1] - (params.momentum2[1]-(params.momentum2[0]))*
                     lr_schedule(fun=params.momentFun,var=t,halfLife=params.halfLife,start=0), theano.config.floatX)

            # PERMUTING T1 AND T2 SETS
            if currentBatch == 0:
                np.random.shuffle(train1Perm)
            if params.useT2 and (currentT2Batch == nBatches2 - 1) :
                np.random.shuffle(train2Perm)
                currentT2Batch = 0
            
            ''' 
                Update T1&T2 
            '''
            # Update both
            if params.useT2:                   
                # make batches                
                sampleIndex1 = train1Perm[(currentBatch * params.batchSize1):
                                        ((currentBatch + 1) * (params.batchSize1))]
                sampleIndex2 = train2Perm[(currentT2Batch * params.batchSize2):
                                        ((currentT2Batch + 1) * (params.batchSize2))]                
                
                if (i % params.T1perT2 ==  0) and ( i >= params.triggerT2):
                
                   res = updateT2part1(t2Data[sampleIndex2], t2Label[sampleIndex2], lr1, moment1, 0, 1)                     
                   (c2, y2, debugs) = (res[0], res[1], res[2:])   
                   
                   res = updateT2part2(t1Data[sampleIndex1], t1Label[sampleIndex1], lr1, moment1, lr2, moment2, 1, 0)  
                   (c1, y1, debugs) = (res[0], res[1], res[2:])   
 
                   tempError2 += [1.*sum(t2Label[sampleIndex2] != y2) / params.batchSize2]
                   tempCost2 += [c2]
                   currentT2Batch += 1                       
                   if np.isnan(c1): print 'NANS in part 2!'
                   if np.isnan(c2): print 'NANS in part 1!'

                else:        
                   res = updateT1(t1Data[sampleIndex1], t1Label[sampleIndex1], lr1, moment1, 1, 0)                      
                   (c1, y1, debugs) = (res[0], res[1], res[2:])
                
                tempError1 += [1.*sum(t1Label[sampleIndex1] != y1) / params.batchSize1]                                   
                tempCost1 += [c1]
                if np.isnan(c1): print 'NANS!'
                
            # Update T1 only 
            else: 
                # make batch
                sampleIndex1 = train1Perm[(currentBatch * params.batchSize1):
                                          ((currentBatch + 1) * (params.batchSize1))]
                res = updateT1(t1Data[sampleIndex1], t1Label[sampleIndex1], lr1, moment1, 1, 0)                      
                (c1, y1, debugs) = (res[0], res[1], res[2:])

             
                tempError1 += [1.*sum(t1Label[sampleIndex1] != y1) / params.batchSize1]
                tempCost1 += [c1]
                if np.isnan(c1): print 'NANS', c1


            '''
                Evaluate test, store results, print status.
            '''
            if np.around(currentProgress % (1./params.trackPerEpoch), decimals=4) == 0 \
                    or i == lastUpdate:
                                        
                # batchnorm parameters: estimate for the final model
                if (params.batchNorm and (currentEpoch > 1)) \
                   and ((currentEpoch % params.evaluateTestInterval) == 0 or i == lastUpdate) \
                   and params.testBN != 'lazy':
                       model = update_bn(model, params, evaluateBN, t1Data, t1Label)    
                     
#                # EVALUATE: validation set
#                allVar = evaluate(vData[:2], vData, vLabel[:2], vLabel, 1)
#                cV, yTest, _ , _ = allVar[0], allVar[1], allVar[2], allVar[3], allVar[4:]
#                #cV, yTest = allVar[0], allVar[1]
#                tempVError = 1.*sum(yTest != vLabel) / nVSamples
#                tempVError = 7.; cV = 7.
                       
                ''' 
                    EVALUATE: test set 
                        - in batches of 1000, ow too large to fit on gpu
                        - using dummy input in place of regularized input stream (Th complains ow)
                        - graph = 1, hence BN constants do not depend on regularized input stream (see batchnorm.py)
                '''    
                if params.model == 'mlp': 
                    nTempSamples = 5000       
                else:
                    nTempSamples = 1000                           
                tempError = 0.; tempCost = 0.; batchSizeT = nTestSamples / 10
                if currentEpoch < 0.8*params.maxEpoch:
                    np.random.shuffle(testPerm)
                    tempIndex = testPerm[:nTempSamples]
                    cT, yTest, p, stats = evaluate(testD[tempIndex], testL[tempIndex], 0, 1)
                    tempError = 1.*sum(yTest != testL[tempIndex]) / nTempSamples
                else:                    
                    for j in range(10):
                        tempIndex = testPerm[j*batchSizeT:(j+1)*batchSizeT]
                        cT, yTest, p, stats = evaluate(testD[tempIndex], testL[tempIndex], 0, 1)
                        tempError += 1.*sum(yTest != testL[tempIndex]) / batchSizeT
                        tempCost += cT
                    tempError /= 10.                     
                    cT = tempCost / 10.

                                                       
                ''' 
                    TRACK: class errors & cost
                '''    
                # note: T1 and T2 errors are averaged over training, hence initially can not be compared to valid and test set
                t1Error += [np.mean(tempError1)]; t1Cost += [np.mean(tempCost1)] 
                if params.useT2:
                    t2Error += [np.mean(tempError2)]; t2Cost += [np.mean(tempCost2)]
                testError +=  [tempError]; testCost += [cT]
                penaltyCost += [p]                                        
                #validError += [tempVError]

                # RESET tracked errors
                tempError1 = []; tempCost1 = []
                tempError2 = []; tempCost2 = []
    
                ''' 
                    TRACK: T2 parameter statistics & learning rates                
                '''                
                # monitoring T2 values 
                if params.useT2:
                    trackNoise, trackPenal = t2_extract(model, params, trackNoise, trackPenal)                    

                # monitoring activations
                if params.trackStats:
                    trackLayers = stat_extract(stats, params, trackLayers)
                
                # monitoring gradients
                if params.trackGrads:
                    trackGrads = grad_extract(debugs, params, sharedNames, trackGrads)         
                        
                # monitoring log learning rates        
                trackLR1 += [lr1]
                trackLR2 += [lr2]
    
                ''' 
                    STATUS print               
                '''                
                if params.useT2 and ((currentEpoch % params.printInterval) == 0 or 
                                     (i == params.maxEpoch*nBatches1 - 1)):
                    print currentEpoch, ') time=%.f     T1 | T2 | test | penalty ' % ((time() - t_start)/60)
                    
                    print 'ERR    %.3f | %.3f | %.3f | - ' % (
                        t1Error[-1]*100, t2Error[-1]*100, testError[-1]*100)
                    print 'COSTS   %.3f | %.3f | %.3f | %.3f ' % (
                        t1Cost[-1], t2Cost[-1], testCost[-1], penaltyCost[-1])

                    print 'Log[learningRates] ', np.log10(lr1), 'T1 ', np.log10(lr2), 'T2'                        
                    for param in params.rglrzTrain:
                        if param in penalList:
                            print param, trackPenal[param][-1]
                        if param in noiseList:
                            print param, trackNoise[param][-1]

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
       allCosts = np.concatenate(([t1Cost], [t2Cost], [testCost], [penaltyCost]), axis = 0) 

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