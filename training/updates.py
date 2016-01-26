import numpy as np
import theano
import theano.tensor as T

from training.adaptive import adam

def updates(mlp, params, globalLR1, globalLR2, momentParam1, momentParam2, phase):
    
    ''' Computing updates of T1 and T2 parameters.
    
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
    epsilon = np.asarray(0.0, dtype=theano.config.floatX)

    # gradients
    cost1 = mlp.classError1 + mlp.penalty
    cost2 = mlp.classError2

    # dC1/dT1, dC2/dT1m dC2/dT2
    gradC1T1 = T.grad(cost1, mlp.paramsT1)
    gradC2T1 = T.grad(cost2, mlp.paramsT1)
    gradC1T2 = T.grad(cost1, mlp.paramsT2)        
    
    # take opt from Adam?
    if params.opt1 in ['adam']:
        opt1 = adam()
    else:
        opt1 = None
    if params.opt2 in ['adam']:
        opt2 = adam()
    else:
        opt2 = None    
        

    def update_fun(param, grad, penaltyparam, dataset, history, opt, params):

        ''' Computing the update from gradient. 
        Adaptive step sizes, learning rate, momentum etc. 
    
        '''

        def separateLR(params, sharedName, globalLR1, globalLR2):

            sharedName, _ = sharedName.split('_')
            customizedLR = globalLR2

            if (sharedName in params.rglrzLR.keys()):
                customizedLR = globalLR2*params.rglrzLR[sharedName]
            return customizedLR      

        # specification of learning rate, (hyper)parameter-type specific
        assert dataset in ['T1', 'T2']
        lr = globalLR1 if dataset == 'T1' else separateLR(params, param.name, globalLR1, globalLR2) 
 
        # no adam
        if opt is None:
            updates = []
            if params.trackGrads:
                old_grad = theano.shared(np.asarray(param.get_value() * 0., dtype='float32'),
                                    broadcastable=param.broadcastable,
                                    name='oldgrad_%s' % param.name)
                updates += [(old_grad, grad)]
                grad_mean = T.mean(T.sqrt(grad**2))
                grad_rel = T.mean(T.sqrt((grad/(param+1e-12))**2))
                grad_angle = T.sum(grad*old_grad)/(T.sqrt(T.sum(grad**2))*T.sqrt(T.sum(old_grad**2))+1e-12) 
                check = T.stacklists([grad_mean, grad_rel, grad_angle])
                other = [grad]
            else:    
                check = []
                other = [grad]
                          
            up = - lr * grad
        # adam update    
        else:
            up, updates, check, other = opt.up(param, grad, params, lr=lr, dataset=dataset)

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

        # min value  |  NOTE assumption: all hyperparams can only be positive
        if dataset == 'T2':
            newparam = T.maximum(epsilon, newparam)

        updates += [(param, newparam)]
        paramUpPair = [(param, check)]
        adamGrad = [other]

        return updates, paramUpPair, adamGrad
        
    
    updateT1 = [] if opt1 is None else opt1.initial_updates()
    updateT2 = [] if opt2 is None else opt2.initial_updates() 
    gradC2T2 = []; tempUps = []

    onlyT1param = []; onlyT2param = []
    upnormdiff = 0  

    history = {'grad': dict(), 'up': dict()}
    historyC2 = {'grad': dict(), 'up': dict()}

    # if gradient dC2/dT1 is also approximated with adam
    if params.avC2grad in ['adam', 'momentum'] and params.useT2:
        
            if params.avC2grad == 'adam': opt3 = adam()
            else: opt3 = None
            tempUps = [] if opt3 is None else opt3.initial_updates()
    
            newC2 = []
            for param, grad in zip(mlp.paramsT1, gradC2T1):            
                tempUp, _, newGrad = update_fun(param, T.reshape(grad, param.shape), None, 'T1', historyC2, opt3, params)
                tempUps += tempUp[:-1]
                newC2 += newGrad
            gradC2T1 = newC2
            
        
    # UPDATE T1 params
    for param, grad in zip(mlp.paramsT1, gradC1T1):                
            ups, pair, _ = update_fun(param, grad, None,
                                        'T1', history, opt1, params)
            updateT1 += ups
            onlyT1param += pair
    
    # UPDATING T2 params
    if params.useT2:        
         
        assert len(mlp.paramsT1) == len(gradC2T1) 
        assert len(mlp.paramsT2) == len(gradC1T2)
        assert len(gradC1T1) == len(gradC2T1)
        # computing hyper-gradient 
        minus_gradC1T1 = map(lambda grad: -grad, gradC1T1)
        gradC2T2 = T.Lop(minus_gradC1T1, mlp.paramsT2, gradC2T1) 
            
        for param, grad in zip(mlp.paramsT2, gradC2T2):
            tempUp, tempPair, _ = update_fun(param, T.reshape(grad, param.shape), None,
                                  'T2', {}, opt2, params)
            updateT2 += tempUp
            onlyT2param += tempPair        
                         
    # monitored variables
    if params.useT2: track2 = [check for (_, check) in onlyT2param] 
    track1 = [check for (_, check) in onlyT1param]  
    debugs = []
    
    if (not params.useT2) and params.trackGrads:
        debugs = track1
    elif params.trackGrads:
        debugs = track1 + track2    

    print "Parameters ",
    print ", ".join([p.name for p in mlp.paramsT2]),
    print "are trained on T2"

    return updateT1, updateT2+tempUps, upnormdiff, debugs
    
    
