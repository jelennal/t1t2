import cPickle
import gzip
import os

from sklearn import preprocessing
import numpy as np
from numpy.random import RandomState    
        
def store(item, name):
    
    ''' Pickle item under name.
    
    '''
    import pickle
    file = open(name+'.pkl','wb')
    pickle.dump(item, file)
    file.close()
    return
    
def permute(data, label, params):

    ''' Permute data.
    
    '''
    rndSeed = RandomState(params.seed)
    permute = rndSeed.permutation(data.shape[0])
    data = data[permute]
    label = label[permute]

    return (data, label)
    

def read(params):

    ''' Read data from 'datasets/...'
    
    '''
    if params.dataset == 'mnist':
        
       filename = 'datasets/mnist.pkl.gz' 
       if not os.path.exists(filename):
           raise Exception("Dataset not found!")
    
       data = cPickle.load(gzip.open(filename))
       t1Data, t1Label = data[0][0], np.int32(data[0][1])
       vData, vLabel = data[1][0], np.int32(data[1][1])
       testD, testL = data[2][0], np.int32(data[2][1])
    
    if params.dataset == 'cifar10':
    
       folderName = 'datasets/cifar-10-batches-py/' # assumes unzipped
       if not os.path.exists(folderName):
           raise Exception("Dataset not found!")
    
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

    if params.dataset == 'not_mnist':
        
       filename = 'datasets/not_mnist.pkl.gz' 
       if not os.path.exists(filename):
           raise Exception("Dataset not found!")
    
       data = cPickle.load(gzip.open(filename))
       t1Data, t1Label = data[0][0], np.int32(data[0][1])
       testD, testL = data[1][0], np.int32(data[1][1])
                      
       split = 400000
       t1Data, t1Label = permute(t1Data, t1Label, params)                
       vData, vLabel = t1Data[split:], t1Label[split:]
       t1Data, t1Label = t1Data[:split], t1Label[:split]

#    TODO
#    elif params.daaset == 'svhn':        
    return  t1Data, t1Label, vData, vLabel, testD, testL



def gcn(data, params):
    
    ''' Global contrast normalization of data. 
    
    '''    
    test = data[0]; rest = data[1:]
    testMean = np.mean(test)        
    testStd = np.std(test)
    print testMean, testStd
    
    temp = []
    for item in [test]+rest:
        temp += [(item-testMean)/testStd]   

    return temp


def zca_white(data, params, eps=1e-5):
    
    ''' ZCA whitening of data.
            
    '''
    test = data[0] 
        
    m = np.mean(test, axis = 0)
    ctest = test -  m    
    covMatrix = np.dot(ctest.T, ctest) / 1.*test.shape[1]
    
    U,S,V = np.linalg.svd(covMatrix)    
    S = np.diag(S)
    ZCA = np.dot(np.dot(U, 1.0/np.sqrt(S + eps)), U.T)
 
    whiteData = []
    for item in data:
        whiteData += [np.dot(item - m, ZCA)] # whitened
    store(ZCA, params.dataset+'_test_zca')

    return whiteData  


def show_samples(samples, nShow):   
    
    ''' Print some input samples.
        
    '''
    import math
    import matplotlib.pyplot as plt
    
    _, nFeatures, x, y = samples.shape
    nColumns = int(math.ceil(nShow/5.))
    for i in range(nShow):
        plt.subplot(5, nColumns, i+1)
        image = samples[i]
        image = np.rollaxis(image, 0, 3); 
        plt.imshow(image) 
        plt.axis('off')


def read_preprocess(params):

    ''' Read data, form T1 and T2 sets, preprocess data.

    '''

    ratioT2 = params.ratioT2
    ratioValid = params.ratioValid
    preProcess = params.preProcess
    preContrast = params.preContrast
    sigmoid = lambda x: 1./(1.+ np.exp(-x))
    
    # read data
    t1Data, t1Label, vData, vLabel, testD, testL = read(params)

    # permuting data    
    vData, vLabel = permute(vData, vLabel, params)
    t1Data, t1Label = permute(t1Data, t1Label, params)

    # form datasets T1 and T2 
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

    # global contrast normalization and ZCA    
    if preProcess in  ['global_contrast_norm', 'global_contrast_norm+zca', 'zca']:
        
        if not params.useT2: t2Data = t1Data[:5, :]
        data = [t1Data, t2Data, testD, vData]        

        if preProcess != 'zca':
            t1Data, t2Data, testD, vData = gcn(data, params)
                        
        if params.dataset == 'cifar10' and preProcess in ['global_contrast_norm+zca', 'zca']:
            data = [t1Data, t2Data, testD, vData] 
            t1Data, t2Data, testD, vData = zca_white(data, params)

    # other kinds of preprocessing            
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

    # contrast 
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


    # new data statistics 
    print ' -data max, min, std'
    print np.max(t1Data), np.min(t1Data)
    print max(np.std(t1Data, axis = 0)), min(np.std(t1Data, axis = 0))
    if params.useT2:
        print np.max(t2Data), np.min(t2Data)        
        print np.max(np.std(t2Data, axis = 0)), np.min(np.std(t2Data, axis = 0))

    print ' -data max, min'

    print '- size T1, valid, T2'
    print t1Data.shape, vData.shape
    if params.useT2: print t2Data.shape
        
    if params.useT2 and params.T2isT1:
        nSamples = t2Data.shape[0]
        t2Data = t1Data[:nSamples]
        t2Label = t1Label[:nSamples]


    # reshape if convnet
    if params.model == 'convnet':
        if params.dataset in ['mnist', 'not_mnist']:
            t1Data = t1Data.reshape(-1, 1, 28, 28)
            vData  =  vData.reshape(-1, 1, 28, 28)
            testD  =  testD.reshape(-1, 1, 28, 28)
            if params.useT2: 
                t2Data = t2Data.reshape(-1, 1, 28, 28)    
            
        if params.dataset in ['cifar10', 'svhn']:
            t1Data = t1Data.reshape(-1, 3, 32, 32)
            vData  =  vData.reshape(-1, 3, 32, 32)
            testD  =  testD.reshape(-1, 3, 32, 32)
            if params.useT2: 
                t2Data = t2Data.reshape(-1, 3, 32, 32)    
            
#        show_samples(t1Data[:100], 50)    
        
    return t1Data, t1Label, t2Data, t2Label, vData, vLabel, testD, testL

