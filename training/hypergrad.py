import theano.tensor as T

penalList = ['L1', 'L2', 'Lmax', 'LmaxSlope', 'LmaxCutoff', 'LmaxHard']
noizList = ['addNoise', 'inputNoise']

def hypergrad(paramsT1, paramsT2, gradC2T1, c1, c2, p1=0., p2=0.):
    

    ''' Function defining the hypergradients: gradients of validation cost 
        with respect to various hyperparameters.     
    
        The function is separating penalty hyperparameters 
        (which is assumed to depend only on W) from noise and other hyperparameters,
        due to otherwise dependancy errors in the Lop operator.
        
        Inputs: 
        
        paramsT1, paramsT2 :: T1 and T2 parameters
        c1, c2 :: cross-entropy on training and validation set
        p1, p2 :: penalty terms on training and validation set (p2 assumed 0)
        
    '''

    rglrzPenal = []; rglrzNoiz = []; gradPenal = []; gradNoiz = []
    W = []; gradC2W = []


    # separate different types of parameters
    for rglrz in paramsT2:
        rglrzType, _ = rglrz.name.split('_')
        if rglrzType in penalList:
            rglrzPenal += [rglrz]                
        elif rglrzType in noizList:
            rglrzNoiz += [rglrz]
        else:
            print 'Hypergradient not implemented for ', rglrzType

    # separate weight parameters and gradients
    for (param, grad) in zip(paramsT1, gradC2T1):
        paramType, _ = param.name.split('_')
        if paramType == 'W':
            W += [param]
            gradC2W += [grad]
                     
    # hyper-gradients        
    if rglrzPenal != []:
        gradPW = T.grad(p1, W)        
        gradPW = [-grad for grad in gradPW]        
        gradPenal = T.Lop(gradPW, rglrzPenal, gradC2W)             
    if rglrzNoiz != []:    
        gradE1T1 = T.grad(c1, paramsT1)
        gradE1T1 = [-grad for grad in gradE1T1]        
        gradNoiz = T.Lop(gradE1T1, rglrzNoiz, gradC2T1)             

    # outputs     
    paramsT2 = rglrzPenal+rglrzNoiz
    gradC2T2 = gradPenal+gradNoiz

    return paramsT2, gradC2T2


