import numpy as np
import theano
import theano.tensor as T

from training.adaptive import adam
from training.hypergrad import hypergrad
from training.monitor import grad_monitor


def remove_nans(x):
    return T.switch(T.isnan(x) + T.isinf(x), 0, x)

def scale_norm(x, threshold=3.):
    norm = T.sqrt(T.sum(x*x))
    multiplier = T.switch(norm < threshold, 1, threshold / norm)
    return x * multiplier

def clip_grad(x, threshold=10.):
    x = T.minimum(x, threshold)
    return x


def separateLR(params, sharedName, globalLR1, globalLR2):    
    ''' 
        Get learning rate from the name of the shared variable.    
    '''
    sharedName, _ = sharedName.split('_')
    customizedLR = globalLR1
    if (sharedName in params.rglrzLR.keys()):   
        customizedLR = globalLR2*params.rglrzLR[sharedName]

    return customizedLR      


def update_fun(param, grad, dataset, history, opt, learnParams, params):
    ''' 
        Computing the update from gradient. 
        Adaptive step sizes, learning rate, momentum etc. 
    '''
    epsilon = np.asarray(0.0, dtype=theano.config.floatX)

    # specification of learning rate, (hyper)param specific
    globalLR1, globalLR2, momentParam1, momentParam2 = learnParams
    assert dataset in ['T1', 'T2']
    lr = globalLR1 if dataset == 'T1' else separateLR(params, param.name, globalLR1, globalLR2) 
 
    # update without adam
    if opt is None:
        updates = []
        if params.trackGrads:
            updates, trackGrads = grad_monitor(param, grad, updates, params, opt)
            other = [grad]
        else:    
            trackGrads = []
            other = [grad]                          
        up = - lr * grad

    # update with adam    
    else:
        up, updates, trackGrads, other = opt.up(param, grad, params, lr, dataset)

    # dictionary param to grad (first time around)
    if params.useT2 and dataset == 'T1':
        history['grad'][param] = grad
        history['up'][param] = up

    # momentum
    if params.use_momentum:
        oldup = theano.shared(np.asarray(param.get_value() * 0., dtype='float32'),
                              broadcastable=param.broadcastable,
                              name='oldup_%s' % param.name)
        momentParam = momentParam1 if dataset == 'T1' else momentParam2
        up += momentParam * oldup
        updates += [(oldup, up)]

    # new parameter
    newparam = param + up

    # min value (assumption: all hyperparams >= 0)
    if dataset == 'T2':
        newparam = T.maximum(epsilon, newparam)

    updates += [(param, newparam)]
    adamGrad = [other]
    return updates, trackGrads, adamGrad


def updates(mlp, params, globalLR1, globalLR2, momentParam1, momentParam2):
    
    ''' 
        Computing updates of T1 and T2 parameters.
    
    Inputs:
        mlp :: model
        params :: specification of the model and training
        globalLR1, globalLR2 :: global learning rates for T1 and T2
        momentParam1, momentParam2 :: momentum parameters for T1 and T2
        phase :: external parameter in case of ifelse (currently not in use)        
        
    Outputs:
        updateT1 :: update of T1 parameters and related shared variables                        
        updateT2 :: update of T2 parameters and related shared variables 
        upnormdiff, debugs :: variable tracked for debugging
                        
    '''    

    # gradients
    cost1 = mlp.trainCost + mlp.penalty
    cost2 = mlp.trainCost

    # dC1/dT1
    gradC1T1 = T.grad(cost1, mlp.paramsT1)
    gradC2T1temp = T.grad(cost2, mlp.paramsT1)
        
    # initialzations    
    opt1 = adam() if params.opt1 in ['adam'] else None
    opt2 = adam() if params.opt2 in ['adam'] else None
    updateT1 = [] if opt1 is None else opt1.initial_updates()
    updateT2 = [] if opt2 is None else opt2.initial_updates() 

    updateC2grad = []; gradC2T1 = []; gradC2T2 = []; tempUps = []
    trackT1grads = []; trackT2grads = []
    history = {'grad': dict(), 'up': dict()}
    historyC2 = {'grad': dict(), 'up': dict()}

    learnParams = [globalLR1, globalLR2, momentParam1, momentParam2]

                   
    ''' 
        Updating T1 params
    '''
    for param, grad in zip(mlp.paramsT1, gradC1T1):                

            grad = scale_norm(remove_nans(grad), threshold=3.)                
            ups, track, _ = update_fun(param, grad, 'T1',
                                       history, opt1, learnParams, params)
            updateT1 += ups
            trackT1grads += [track]



    ''' 
        Updating T2 params
    '''

    if params.useT2:     


        '''
            Save grads C2T1 for the T2 update:
        '''
        for param, grad in zip(mlp.paramsT1, gradC2T1temp):   

                grad = scale_norm(remove_nans(grad), threshold=3.)         
                grad = clip_grad(grad, threshold=10.)                
                saveGrad = theano.shared(np.asarray(param.get_value() * 0., dtype='float32'),
                                         broadcastable=param.broadcastable,
                                         name='gradC2T1_%s' % param.name)
                updateC2grad += [(saveGrad, grad)]                         
                gradC2T1 += [saveGrad]

        ''' 
            If gradient dC2/dT1 is also estimated with adam
        '''        
        if params.avC2grad in ['adam', 'momentum']:
                #gradC2T1 = T.grad(cost2, mlp.paramsT1)
                if params.avC2grad == 'adam': opt3 = adam()
                else: opt3 = None
                tempUps = [] if opt3 is None else opt3.initial_updates()
        
                newC2 = []
                grad = scale_norm(remove_nans(grad), threshold=3.)                                
                grad = clip_grad(grad, threshold=10.)                
                for param, grad in zip(mlp.paramsT1, gradC2T1):            
                    tempUp, _, newGrad = update_fun(param, T.reshape(grad, param.shape), 'T1', 
                                                    historyC2, opt3, learnParams, params)
                    tempUps += tempUp[:-1]
                    newC2 += newGrad
                gradC2T1 = newC2
                
        
        paramsT2, gradC2T2 = hypergrad(mlp.paramsT1, mlp.paramsT2, gradC2T1, 
                                       mlp.trainCost, mlp.trainCost, mlp.penalty)            

        for param, grad in zip(mlp.paramsT2, gradC2T2):
            paramName, _ = param.name.split('_')
            if params.decayT2 > 0. and paramName not in ['L2', 'L1']:
                grad += params.decayT2*param 

            grad = scale_norm(remove_nans(grad), threshold=3.) 
            grad = clip_grad(grad, threshold=10.)                              
            tempUp, track, _ = update_fun(param, T.reshape(grad, param.shape),'T2',
                                          {}, opt2, learnParams, params)
            updateT2 += tempUp
            trackT2grads += [track]       
                         
    # monitored variables for output                         
    if (not params.useT2) and params.trackGrads:
        debugs = trackT1grads
    elif params.trackGrads:
        debugs = trackT1grads + trackT2grads    
    else:
        debugs = []
    print "Parameters ",
    print ", ".join([p.name for p in mlp.paramsT2]),
    print "are trained on T2"

    return updateT1, updateT2+tempUps, updateC2grad, debugs
    
    
