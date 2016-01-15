import math

def lr_schedule(fun, var, halfLife, start):
    
    ''' Implementation of various learning rate schedules.
    
    Arguments:
        fun        - which learning rate schedule       
        var        - current position in the total training, var = current update / total updates
        halfLife   - halfLife parameter for exponential function 
        start      - parameter for functions which are zero until start   
        
    '''
    

    def Trapz(t):
        t1 = 1./2 # grows
        t2 = 1./4 # 1
        t3 = 1.-(t1+t2) # falls

        if t < t1:return t/t1
        elif t < (t2+t1): return 1.
        else: return (1. - (t-(1.-t3)) / t3)

    def ZeroTrapz(t):
        t1 = 1./8 # zero
        t2 = 3./8 # grows
        t3 = 1./4 # 1
        t4 = 1.-(t3+t2+t1) # falls

        if t < t1: return 0.
        elif t < (t2+t1): return (t-t1)/t2
        elif t < (t3+t2+t1): return 1.
        else: return (1. - (t-(1.-t4)) / t4)

    def HardZeroOneZero(t):
        t1 = 1./6 # grows
        t2 = 2./3 # 1
#        t3 = 1.-(t1+t2) # falls
        if t < t1: return 0.
        elif t < (t2+t1): return 1.
        else: return 0.
   
    def ThreePhase(t):
        t1 = 1./3
        t2 = 2./3
        if t<t1: return 1.
        elif t>t2: return 0.01
        else: return 0.1    


    Nothing = lambda t: 1.
    LinearVulgaris = lambda t: (1. - t)
    OneLinear = lambda t: min(1, 1 - 3*(t-2./3))#(t - halfLife)/(1-halfLife)) 
    ZeroLinear = lambda t: max(0, (t-start)/(1-start)) 
    Exponential = lambda t: math.pow(2,-t/halfLife)
    InverseT = lambda t: 1./(1.+t/halfLife)
    ZeroOneZero = lambda t: math.sin(t*math.pi)**2
    ZeroThenOne = lambda t: math.ceil(max(0, (t-halfLife)/(1-halfLife)))
    Periodic = lambda t: max(math.sin(-10*t*math.pi), 0)


    function = {
        'None': Nothing,
        'lin': LinearVulgaris,
        'olin': OneLinear,
        'zlin': ZeroLinear,
        'trapezoid': Trapz,
        'ztrapezoid': ZeroTrapz,
        'exp': Exponential,
        'inverse_t': InverseT,
        'zoz': ZeroOneZero,
        'step': ZeroThenOne,
        'barier': HardZeroOneZero,
        'period': Periodic,
        'exp3step': ThreePhase
    }[fun]

    return function(var)



