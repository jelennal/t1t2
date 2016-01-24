import numpy as np
import theano
import theano.tensor as T

from training.adaptive import adam

# ----------------------------------------------------------------------------------- 
def updates(mlp, params, globalLR1, globalLR2, momentParam1, momentParam2, phase):
    jacCost = params.train_T2_gradient_jacobian
    # small number approximating 0
    epsilon = np.asarray(0.0, dtype=theano.config.floatX)

    # gradient of T1 ----------------------------------- GRADS
    cost1 = mlp.classError1 + mlp.penalty
    gradT1 = T.grad(cost1, mlp.paramsT1)

    if jacCost:
        cost2 = mlp.classError2
        # gradients of T2 - c2/w1, c1/lam
        gradC2 = T.grad(cost2, mlp.paramsT1)
        gradT1reg = T.grad(cost1, mlp.paramsT2)
        
    else:
        cost2 = mlp.classError2
        gradC2 = T.grad(cost2, mlp.paramsT1)            
        
        gradT1reg = T.grad(cost1, mlp.paramsT2)
        gradT2exact = []

        for t2 in mlp.paramsT2:
            if params.cost2Type == 'C2-C1':    
               tempGrad = T.grad(cost2, t2)
            else:
               tempGrad = epsilon       
            if tempGrad is None:
                   raise theano.gradient.DisconnectedInputError, 'Cost2 does not' \
                       + ' depend on %s' % t2.name
                   tempGrad = epsilon       
            gradT2exact += [tempGrad]
#       gradT2exact = T.grad(cost2, mlp.paramsT2)

    # take opt from Adam?
    if params.opt1 in ['adam']:
        opt1 = adam()
    else:
        opt1 = None
    if params.opt2 in ['adam']:
        opt2 = adam()
    else:
        opt2 = None    
        

# -----------------------------------------------------------
    def update_fun(param, grad, penaltyparam, dataset, history, opt, params):

        def separateLR(params, sharedName, globalLR1, globalLR2):
            sharedName, _ = sharedName.split('_')
            customizedLR = globalLR2
            #if params.opt2 == 'adam':
            if (sharedName in params.rglrzLR.keys()):
                customizedLR = globalLR2*params.rglrzLR[sharedName]
            return customizedLR      
 
        assert dataset in ['T1', 'T2']
#        lr = globalLR1 if dataset == 'T1' else globalLR2
        # more detailed determining of learning rate
        lr = globalLR1 if dataset == 'T1' else separateLR(params, param.name, globalLR1, globalLR2) 
 
        # Standard update
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
        else:
            up, updates, check, other = opt.up(param, grad, params, lr=lr, dataset=dataset)

        # dictionary param to grad (first time around)
        if params.useT2 and dataset == 'T1':
            history['grad'][param] = grad
            history['up'][param] = up
        # add momentum to update
        if params.use_momentum:
            oldup = theano.shared(np.asarray(param.get_value() * 0., dtype='float32'),
                                  broadcastable=param.broadcastable,
                                  name='oldup_%s' % param.name)
            momentParam = momentParam1 if dataset == 'T1' else momentParam2
            up += momentParam * oldup
            updates += [(oldup, up)]

        # New parameter
        newparam = param + up
#        # backprojected w cutoff penalty
#        if 'LmaxHard' in params.rglrz and penaltyparam is not None:
#            print "Using maxnorm for parameter %s with value %2.2f" % \
#                (param.name, penaltyparam.get_value())
#            col_L2_norms = T.sqrt(T.sum(T.sqr(newparam), axis=0))
#            norm_divisor = T.clip(col_L2_norms / penaltyparam, 1, np.inf) # is it elementwise?
#            newparam = newparam / norm_divisor
#        # min value  |  NOTE assumption: all hyperparams can only be positive
        if dataset == 'T2':
            newparam = T.maximum(epsilon, newparam)

        updates += [(param, newparam)]
        paramUpPair = [(param, check)]
        adamGrad = [other]

        return updates, paramUpPair, adamGrad
# -----------------------------------------------------------
        
        
    
    # SET UPDATES -----------------------------------------------------------
    updateT1 = [] if opt1 is None else opt1.initial_updates()
    updateT2 = [] if opt2 is None else opt2.initial_updates()
 
    tempUps = []

    onlyT1param = []
    onlyT2param = []
    gradT1adam = []
    history = {'grad': dict(), 'up': dict()}
    historyC2 = {'grad': dict(), 'up': dict()}

    newParamsT1 = []
    upnormdiff = 0  

    # -------------------------------------------------- new

    if params.avC2grad in ['adam', 'momentum'] and params.useT2:
            if params.avC2grad == 'adam': opt3 = adam()
            else: opt3 = None
            tempUps = [] if opt3 is None else opt3.initial_updates()
    
            newC2 = []
            for param, grad in zip(mlp.paramsT1, gradC2):            
                tempUp, _, newGrad = update_fun(param, T.reshape(grad, param.shape), None, 'T1', historyC2, opt3, params)
                tempUps += tempUp[:-1]
                newC2 += newGrad
            gradC2 = newC2
            
        
        # UPDATING T1
    for param, grad in zip(mlp.paramsT1, gradT1):
            # ---------------------------------------------------------------------- CHECK THIS LINE
            if params.checkROP:
                grad = T.Rop(grad, param, grad)
                # T.Lop(cost1, param, T.Rop(cost1, param, grad))                
                
            tempUp, tempPair, tempAdam = update_fun(param, grad, None,#mlp.penaltyMaxParams.get(param, None),
                                        'T1', history, opt1, params)
            updateT1 += tempUp
            onlyT1param += tempPair
            gradT1adam += tempAdam # [grad]

    
    if params.useT2:
        gradreg = [] 
        assert len(mlp.paramsT1) == len(gradC2) 
        assert len(mlp.paramsT2) == len(gradT1reg)
        if not jacCost:
            assert len(mlp.paramsT2) == len(gradT2exact)    
        
        # UPDATING T2 ------------------------------- DIY
        
#        rop_grad_noise = []; rop_grad_s = []; rop_s = []
#        
#        for (i, grad) in zip(range(params.nLayers - 1), gradC2):
#            rop_s = g*mlp.h[i-1] rop_s
        
        
        if not jacCost:
            
            
#            for (t2, t2e) in zip(mlp.paramsT2, gradT2exact):
#                    accumulate = None
#                    for (gt1, gt2) in zip(gradT1, gradC2):
#                        try:
#                            val = T.Lop(-gt1, t2, gt2)
#                        except theano.gradient.DisconnectedInputError:
#                            val = None
#                        accumulate = val if accumulate is None else (
#                            accumulate if val is None else accumulate + val)
#                    if accumulate is None:
#                        raise theano.gradient.DisconnectedInputError, 'Cost does not' \
#                            + ' depend on %s' % t2.name
#                    gradreg += [accumulate + t2e]

            assert len(gradT1) == len(gradC2)
            print len(gradT1), len(mlp.paramsT2), len(gradC2)
            gradreg = T.Lop(gradT1, mlp.paramsT2, gradC2) # add - to each
            print len(gradreg)           
            
        else:
                                
            for (t2, g1r) in zip(mlp.paramsT2, gradT1reg):
                accumulate = None
                for (t1, gt2) in zip(mlp.paramsT1, gradC2):
                    try:
 #                       val = T.Rop(-g1r, t1, gt2)
                        val = T.Rop(-g1r.flatten(), t1, gt2)
                    except theano.gradient.DisconnectedInputError:
                        # print "%s does not affect %s" % (t2.name, t1.name)
                        val = None
                    accumulate = val if accumulate is None else (
                        accumulate if val is None else accumulate + val)
                if accumulate is None:
                    raise theano.gradient.DisconnectedInputError, 'Cost does not' \
                        + ' depend on %s' % t2.name
                gradreg += [accumulate]

#            assert len(mlp.paramsT1) == len(gradC2)
#            gradreg = T.Rop(gradT1reg, mlp.paramsT1, gradC2) # add - to each
#            print len(gradreg)           
                                
   
        for param, grad in zip(mlp.paramsT2, gradreg):
            if params.T2onlySGN:
                grad = T.sgn(grad)
            tempUp, tempPair, _ = update_fun(param, T.reshape(grad, param.shape), None,
                                  'T2', {}, opt2, params)
            updateT2 += tempUp
            onlyT2param += tempPair        
            
             
    
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

    return updateT1, updateT2+tempUps, upnormdiff, debugs#, T2_grads
    
    
