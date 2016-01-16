import numpy as np
import theano
import theano.tensor as T

'''
    Functions for monitoring training. 
    
    grad_monitor - monitors gradient/update norms, angle between gradients/updates; 
    stat_monitor - monitors network statistics
    t2_monitor - monitors t2_parameters
    
'''

def grad_monitor(param, grad, updates, g_t, m, v, e, params):
    
    zero = np.float32(0.)
    old_grad = theano.shared(np.float32(param.get_value()) * zero, name="old_grad_%s" % param.name)
    updates.append((old_grad, grad))
    old_g_t = m/(T.sqrt(v) + e) 
    all_grads = {
        'grad' : T.mean(T.sqrt(grad**2)),
        'grad_rel' : T.mean(T.sqrt((grad/(param+1e-12))**2)),
        'grad_angle' : T.sum(grad*old_grad)/(T.sqrt(T.sum(grad**2))*T.sqrt(T.sum(old_grad**2))+1e-12) ,
        'grad_max' : T.max(T.sqrt(grad**2)),
        'p_t' : T.mean(T.sqrt((g_t)**2)),
        'p_t_rel' : T.mean(T.sqrt((g_t/(param+1e-12))**2)),
        'p_t_angle' : T.sum(g_t*old_g_t)/(T.sqrt(T.sum(g_t**2))*T.sqrt(T.sum(old_g_t**2)+1e-12)),
        'p_t_max' : T.max(T.sqrt(grad**2))
        }
    
    if params.whichGrads == 'all':
        temp = []
        for grad_type in params.listGrads:
            temp += [all_grads[grad_type]] 
        check = T.stacklists(temp)
    else:
        check = all_grads[params.whichGrads]
    
    return updates, check    


def stat_monitor(layers, params):
    
    i=0
    eps = 1e-4
    netStats =  {}
    for key in params.activTrack:
        netStats[key] =  []

    for layer in layers:
        
        output = layer.output
        if params.mode == 'convnet' and params.convLayers[i].type != 'conv':
            W=0.; b=0.; a=0.  
        else:        
            W = layer.W; b=layer.b; a=layer.a
        
        if params.model == 'convnet':
            tempMean = T.mean(output, axis = (0, 2, 3))
            tempSTD = T.std(output, axis = (0, 2, 3))
            tempMax = T.max(output, axis = (0, 2, 3))
        else:    
            tempMean = T.mean(output, axis = 0)
            tempSTD = T.std(output, axis = 0)
            tempMax = T.max(output, axis = 0)
        
        tempSpars = T.mean(T.le(input, eps))                
        if (key == 'rnoise' or key == 'rnstd') and 'addNoise' in params.rglrz and 'addNoise' not in params.rglrzPerMap:
             tempRNoise = layer.rglrzParam['addNoise']/tempSTD
        else:
             tempRNoise = 0.       
        
        i += 1
        for key in params.activTrack:
            statistic = {'mean': T.mean(tempMean),
                         'std': T.mean(tempSTD),
                         'max': T.max(tempMax),
                       'const': T.mean(T.le(tempSTD, eps)), 
                       'spars': tempSpars,
                       'wmean': T.mean(abs(W)),
                        'wstd': T.std(W),
                        'wmax': T.max(abs(W)),
                      'rnoise': T.mean(tempRNoise),
                       'rnstd': T.std(tempRNoise),
                       'bias' : T.mean(b),
                           'a': T.mean(a),
                        'bstd': T.std(b),
                        'astd': T.std(a),
                       }[key]
            netStats[key] +=  [statistic]
                   
    allStats = [] 
    if params.trackStat:
#        for key in params.activTrack: allStats += [netStats[key]]
#        hStat = T.stacklists(allStats)
        hStat = netStats
    else: 
        hStat =  0.

    
    return hStat