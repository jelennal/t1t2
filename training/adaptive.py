import numpy as np
import theano
import theano.tensor as T

def gradient_tracking(param, grad, updates, g_t, m, v, e, params):
    
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
                


class adam(object):
    ''' Adam adaptive step size + gradient tracking
    
    # Arguments: 
        param -- trained parameter*, 
        grad -- gradient with respect to param*, 
        params -- all model parameters
        lr -- param*'s learning rate
        dataset -- param* is elementary (T1) or hyperparameter (T2)
    '''    

    def __init__(self, b1=0.9, b2=0.999, e=1e-8, lam=(1-1e-8), name='adam'):         
        self.b1 = np.float32(b1)
        self.b2 = np.float32(b2)
        self.e = np.float32(e)
        self.lam = np.float32(lam)
        self.i = theano.shared(np.float32(1.), name=name)

    def initial_updates(self):        
        return [(self.i, self.i + 1.)]

    def up(self, param, grad, params, lr=1e-4, dataset='T1'):
        zero = np.float32(0.)
        one = np.float32(1.)
        updates = []
        check = []
        other = []
      
        # initialize adam shared variables
        m = theano.shared(np.float32(param.get_value()) * zero, name="m_%s" % param.name)
        v = theano.shared(np.float32(param.get_value()) * zero, name="v_%s" % param.name)

        fix1 = one - self.b1 ** self.i
        fix2 = one - self.b2 ** self.i
        b1_t = self.b1 * self.lam ** (self.i - 1)
        
        lr_t = lr * (T.sqrt(fix2) / fix1)        
        m_t = ((one - b1_t) * grad) + (b1_t * m)
 #       m_t = ((one - self.b1) * grad) + (self.b1 * m)
        v_t = ((one - self.b2) * T.sqr(grad)) + (self.b2 * v)
        g_t = m_t / (T.sqrt(v_t) + self.e)
        p_t = - (lr_t * g_t)

        # update Adam shared variables
        updates.append((m, m_t))
        updates.append((v, v_t))

        # in case of gradient tracking
        if params.trackGrads:
            updates, check = gradient_tracking(param, grad, updates, g_t, m, v, self.e, params)

        # if approximationg gradC2 with adam
        if params.avC2grad in ['adam', 'momentum']:
            other = g_t * (T.sqrt(fix2) / fix1) # alt: -lr_t*g_t or m_t

 
 
        return p_t, updates, check, other



