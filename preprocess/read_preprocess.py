import cPickle
import gzip

from sklearn import preprocessing
import numpy as np
from numpy.random import RandomState

#import math
#import matplotlib.pyplot as plt
#def ShowSamples(samples, nShow):    
#    _, nFeatures, x, y = samples.shape
#    nColumns = int(math.ceil(nShow/5.))
#    for i in range(nShow):
#        plt.subplot(5, nColumns, i+1)
#        image = samples[i]
#        image = np.rollaxis(image, 0, 3); 
#        plt.imshow(image) 
#        plt.axis('off')
        
def store(item, name):
    import pickle;
    file = open(name+'.pkl','wb');
    pickle.dump(item, file);
    file.close()
    return

def gcn(data, params):

#    temp = []
#    eps = 1e-8
#    dim = data[0].shape
#    for item in data:
#        m = item.mean(axis = 1)[:, np.newaxis]
#        v = item.var(axis = 1)[:, np.newaxis]
##        v[v < eps] = np.float32(1.)
#        print item.shape, m.shape, item.mean(axis = 1).shape
#        item -= m
#        item /= np.sqrt(v+0.001)
#        temp += [item]

    if params.dataset == 'mnist':

        test = data[0]; rest = data[1:]
        testMean = np.mean(test)
        testStd = np.std(test)
        print testMean, testStd
        
        temp = []
        for item in [test]+rest:
            temp += [( item - testMean )/ testStd]
        
    if params.dataset == 'cifar10':

 
        test = data[0]; rest = data[1:]
        testMean = np.mean(test)#, axis = (0, 2))        
        testStd = np.std(test)#, axis = (0, 2)) 
        print testMean, testStd
        
        temp = []
        for item in [test]+rest:
            temp += [(item-testMean)/testStd]   

    return temp


def zca_white(data, params):

    epsilon = 0.00001 
    test = data[0] 
        
    m = np.mean(test, axis = 0)
    ctest = test -  m    
    covMatrix = np.dot(ctest.T, ctest) / 1.*test.shape[1]
    print covMatrix.shape, m.shape
    
    U,S,V = np.linalg.svd(covMatrix)    
    S = np.diag(S)
    ZCA = np.dot(np.dot(U, 1.0/np.sqrt(S + epsilon)), U.T)
 
    whiteData = []
    for item in data:
        print item.shape, ZCA.shape 
        whiteData += [np.dot(item - m, ZCA)] # whitened
        print whiteData[-1].shape

    store(ZCA, params.dataset+'_test_zca')

    return whiteData  


def read_preprocess(params):

    ratioT2 = params.ratioT2
    ratioValid = params.ratioValid
    preProcess = params.preProcess
    preContrast = params.preContrast
    sigmoid = lambda x: 1./(1.+ np.exp(-x))
    
    # which dataset? --------------------------------------------------
    if params.dataset == 'mnist':
       filename = 'datasets/mnist.pkl.gz' 

       data = cPickle.load(gzip.open(filename))

       t1Data, t1Label = data[0][0], np.int32(data[0][1])
       vData, vLabel = data[1][0], np.int32(data[1][1])
       testD, testL = data[2][0], np.int32(data[2][1])

    elif params.dataset == 'cifar10':

       folderName = 'datasets/cifar-10-batches-py/'
       batchNames = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4'] 
       t1Data, t1Label = np.empty((0,3072), dtype = float), np.empty((0), dtype = int)

       for item in batchNames: 
           fo = open(folderName + item, 'rb'); dict = cPickle.load(fo); fo.close()
           t1Data = np.append(t1Data, np.float32(dict['data']), axis = 0)
           t1Label = np.append(t1Label, np.int32(dict['labels']))
           
       fo = open(folderName + 'data_batch_5', 'rb'); dict = cPickle.load(fo); fo.close()
       vData = np.float32(dict['data']); vLabel = np.int32(dict['labels'])  
       fo = open(folderName + 'test_batch', 'rb'); dict = cPickle.load(fo); fo.close()
       testD = np.float32(dict['data']); testL = np.int32(dict['labels'])   
#    FIX LATERZ
#    elif params.daaset == 'svhn':        
        
    # load data -------------------------------------------------------
    

    rndSeed = RandomState(43) # CHECK THIS LATER
    def permute(data, label):
        permute = rndSeed.permutation(data.shape[0])
        data = data[permute]
        label = label[permute]
        return (data, label)

    vData, vLabel = permute(vData, vLabel)
    t1Data, t1Label = permute(t1Data, t1Label)


    # setup the sets T1 and T2 (currently: T1 = 5, T2 = 1)-------------
    if params.useT2:
        nVSamples = vData.shape[0]
        # set up t2+validation
        if ratioT2 > 1.:
            tempIndex = int(round((ratioT2 - 1.)*nVSamples))
            tempData = t1Data[:tempIndex]
            tempLabel = t1Label[:tempIndex]
            vData = np.concatenate((vData, tempData))
            vLabel = np.concatenate((vLabel, tempLabel))
            t1Data = t1Data[tempIndex:]
            t1Label = t1Label[tempIndex:]
        elif ratioT2 < 1.:
            tempIndex = int(round((1.-ratioT2)*nVSamples))
            tempData = vData[:tempIndex]
            tempLabel = vLabel[:tempIndex]
            t1Data = np.concatenate((t1Data, tempData))
            t1Label = np.concatenate((t1Label, tempLabel))
            vData = vData[tempIndex:]
            vLabel = vLabel[tempIndex:]
        # shuffle indices in t2+validation
        nVSamples = vData.shape[0]
        # set up t2 and validation
        if params.useVal:
           tempIndex = int(round(nVSamples*(1.-ratioValid)))
           t2Data = vData[:tempIndex]
           t2Label = vLabel[:tempIndex]
           vData = vData[tempIndex:]
           vLabel = vLabel[tempIndex:]
        else:   
           tempIndex = int(round(nVSamples*(1.-ratioValid)))
           t2Data = vData
           t2Label = vLabel
           vData = vData[tempIndex:]
           vLabel = vLabel[tempIndex:]

    else:
        t2Data = []
        t2Label = [] 
        if not params.useVal:
           t1Data = np.concatenate((vData, t1Data))
           t1Label = np.concatenate((vLabel, t1Label))            

    
    if preProcess in  ['global_contrast_norm', 'global_contrast_norm+zca', 'zca']:

#        if params.dataset == 'cifar10' and preProcess == 'zca':
#            t1Data=t1Data.reshape(-1, 3, 32*32) 
#            t2Data=t2Data.reshape(-1, 3, 32*32) 
#            testD = testD.reshape(-1, 3, 32*32) 
#            vData = vData.reshape(-1, 3, 32*32)
#        
        if not params.useT2: t2Data = t1Data[:5, :]
        data = [t1Data, t2Data, testD, vData]        

        if preProcess != 'zca':
            t1Data, t2Data, testD, vData = gcn(data, params)
                        
#        if params.dataset == 'cifar10' and preProcess in ['global_contrast_norm+zca', 'zca']:
#            for i in range(3):
#                data = [t1Data[:, i, :], t2Data[:, i, :], testD[:, i, :], vData[:, i, :]]
#                t1Data[:, i, :], t2Data[:, i, :], testD[:, i, :], vData[:, i, :] = ZCAWhitening(data, params)
#
#        if params.dataset == 'mnist' and preProcess == 'global_contrast_norm+zca':
        if params.dataset == 'cifar10' and preProcess in ['global_contrast_norm+zca', 'zca']:

            data = [t1Data, t2Data, testD, vData] 
            t1Data, t2Data, testD, vData = zca_white(data, params)

        print '- data mean, std'        
        print np.mean(t1Data), np.std(t1Data), t1Data.shape            
        print np.mean(t2Data), np.std(t2Data), t2Data.shape            
        print np.mean(testD), np.std(testD), testD.shape            
            
    else:         
        scaler = {
             'm0': preprocessing.StandardScaler(with_std = False).fit(t1Data),
             'm0s1': preprocessing.StandardScaler().fit(t1Data),
             'minMax': preprocessing.MinMaxScaler().fit(t1Data),
             'None': 1.
             }[preProcess]
        if preProcess != 'None':
           t1Data = scaler.transform(t1Data)
           if params.useT2: t2Data = scaler.transform(t2Data)
           vData = scaler.transform(vData)
           testD = scaler.transform(testD)

    # contrast ------------------------------------------------------
    contrastFun = {
         'tanh': np.tanh,
         'arcsinh': np.arcsinh,
         'sig': sigmoid,
         'None': 1.
         }[preContrast]
    if preContrast != 'None':
       t1Data = contrastFun(t1Data)
       if params.useT2: t2Data = contrastFun(t2Data)
       vData = contrastFun(vData)
       testD = contrastFun(testD)


    # new data statistics -------------------------------------------
    print ' -data max, min'
    print np.max(t1Data), np.min(t1Data), np.max(t2Data), np.min(t2Data)
    print '- size T1, valid, T2'
    print t1Data.shape, vData.shape
    if params.useT2: print t2Data.shape
        
    if params.useT2 and params.T2isT1:
        nSamples = t2Data.shape[0]
        t2Data = t1Data[:nSamples]
        t2Label = t1Label[:nSamples]

    # ---------- reshape if convnet
    if params.model == 'convnet':
        if params.dataset == 'mnist':
            t1Data = t1Data.reshape(-1, 1, 28, 28)
            vData  =  vData.reshape(-1, 1, 28, 28)
            testD  =  testD.reshape(-1, 1, 28, 28)
            if params.useT2: 
                t2Data = t2Data.reshape(-1, 1, 28, 28)    
            
        if params.dataset == 'cifar10':
            t1Data = t1Data.reshape(-1, 3, 32, 32)
            vData  =  vData.reshape(-1, 3, 32, 32)
            testD  =  testD.reshape(-1, 3, 32, 32)
            if params.useT2: 
                t2Data = t2Data.reshape(-1, 3, 32, 32)    
            
#        ShowSamples(t1Data[:100], 50)    

    print t1Data.shape, vData.shape, testD.shape
    if params.useT2: print t2Data.shape
        


    return t1Data, t1Label, t2Data, t2Label, vData, vLabel, testD, testL


#if not os.path.exists(filename):
#       raise Exception("Dataset not found, please run:\n  wget http://deeplearning.net/data/mnist/mnist.pkl.gz")