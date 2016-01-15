import numpy as np

def conv_setup(params):

    ''' Defining the convolutional architecture.
    
    Arguments: Parameters defining the entire model.        
    
    '''
    
    class conv_layer():
        def __init__(self, layerType, dimFilters, nFilters, doNoise = True, doBN = True):            
            self.type = layerType
            self.filter = dimFilters
            self.maps = nFilters
            self.noise = doNoise
            self.bn = doBN
            
    cl1 = conv_layer('conv', (3, 3), (3, 96))
    cl2 = conv_layer('conv', (3, 3), (96, 96))
    cl3 = conv_layer('pool', (3, 3), (96, 96), 1, 0)
    
    cl4 = conv_layer('conv', (3, 3), (96, 192))
    cl5 = conv_layer('conv', (3, 3), (192, 192))
    cl6 = conv_layer('conv', (3, 3), (192, 192))
    cl7 = conv_layer('pool', (3, 3), (192, 192), 1, 0)
 
    cl8  = conv_layer('conv', (3, 3), (192, 192))
    cl9  = conv_layer('conv', (1, 1), (192, 192))
    cl10 = conv_layer('conv', (1, 1), (192, 10))
    cl11 = conv_layer('average+softmax', (6, 6), (10, 10), 0, 0)

    cl11alt = conv_layer('average', (6, 6), (10, 10))
    cl12alt = conv_layer('softmax', (6, 6), (10, 10))
    
    if params.dataset == 'mnist':
        cl1 = conv_layer('conv', (3, 3), (1, 96))
        
    conv_layers = [cl1, cl2, cl3, cl4, cl5, cl6, cl7, cl8, cl9, cl10, cl11]   
    return conv_layers
    
           

def setup(replace_params={}):
    
    ''' Defining the entire neural network model and methods for training.
    
    Arguments:  Dictionary of the form {'paramName': paramValue}. 
                E.g. replace_params = {'useT2': False, 'learnRate1': 0.1}
    '''
    
    ones = np.ones(20)    
    class Params():
        def __init__(self):
            # T2
            self.useT2 = 1                                                     # use T2? 
            self.useVal = 0                                                    # use validation? 
            self.T1perT2 = 10                                                  # how many T1 updates per one T2 update?
            self.T1perT2epoch = 1                                              # how many epochs with T1 updates per one epoch with T2 updates?
            self.saveName = 'result.pkl'                                       # where to save the data?
            self.T2isT1 = False                                                # sanity check: what if T2 is a subset of T1?
            # MODEL
            self.model = 'convnet'                                             # which model? 'mlp'/'convnet' 
            self.dataset = 'cifar10'                                           # which dataset? 'mnist'/'svhn'/'cifar10'/'cfar100'
            # PREPROCESSING
            self.ratioT2 = 0.5                                                 # how much of validation set goes to T2? [0-1]
            self.ratioValid = 0.05                                             # how much of T2 goes to validatio set
            self.preProcess = 'global_contrast_norm'                           # what input preprocessing? 'None'/'m0'/'m0s1'/'minMax'/'pca'/'global_contrast_norm'/'zca'/'global_contrast_norm+zca'
            self.preContrast = 'None'                                          # nonlinear transform over input? 'None'/'tanh'/'arcsinh'/'sigmoid'
            # ARCHITECTURE
            self.nHidden = [784, 256, 256, 10]                                 # how many hidden units in each layer?
            self.activation = ['relu','relu','softmax']                        # what nonlinearities in each layer?                      
            self.nLayers = len(self.nHidden)-1                                 # how many layers are there? 
            # BATCH NORMALIZATION                                               
            self.batchNorm = True                                              # use batch normalization?
            self.aFix = True                                                   # fix scalling parameter?
            self.movingAvMin = 0.15                                            # moving average paramerer? [0.05-0.20]
            self.movingAvStep = 1                                              # moving average step size? 
            self.evaluateTestInterval = 15                                     # how often compute the "exact" BN parameters? i.e. replacing moving average with the estimate from the whole training data
            self.m = 55                                                        # when computing "exact" BN parameters, average over how many samples from training set?
            self.testBN = 'default'                                            # when computing "exact" BN parameters, how? 'default'/'proper'/'lazy'
            # REGULARIZATION
            self.rglrzTrain = ['addNoise', 'L2']                                     # which rglrz are trained? (which are available? see: rglrzInitial)
            self.rglrz = ['addNoise', 'L2']                                          # which rglrz are used? 
            self.rglrzPerUnit = []                                             # which rglrz are defined per hidden unit? (default: defined one per layer) 
            self.rglrzPerMap = []                                              # which rglrz are defined per map? (for convnets)
            self.rglrzPerNetwork = []                                          # which rglrz are defined per network?
            self.rglrzPerNetwork1 = []                                         # which rglrz are defined per network? BUT have a separate param for the first layer      
            self.rglrzInitial = {'L1': 0.*ones,                                # initial values of rglrz 
                                 'L2': 0.*ones,
                         'LmaxCutoff': 0.*ones,                                # soft cutoff param1
                          'LmaxSlope': 0.*ones,                                # soft cutoff param2
                           'LmaxHard': 2.*ones,                                # hard cutoff aka maxnorm 
                          'addNoise' : 0.*ones, 
                        'inputNoise' : [0.],                                   # only input noise (if trained, need be PerNetwork)
                            'dropOut': [0.2, 0.5, 0.5, 0.5],
                           'dropOutB': [0.2, 0.5, 0.5, 0.5]}                   # shared dropout pattern within batch
            self.rglrzLR = {'L1': 0.00001,                                     # regularizer specific learning rates 
                            'L2': 0.001, 
                    'LmaxCutoff': 0.1, 
                     'LmaxSlope': 0.0001, 
                     'addNoise' : 1., 
                   'inputNoise' : 1.}
            # REGULARIZATION: noise specific                
            self.noiseupSoftmax = False                                        # is there noise in the softmax layer?
            self.noiseWhere = 'type0'                                          # where is noise added at input? 'type0' - after non-linearity, 'type1' - before non-linearity                 
            self.noiseT1 = 'None'                                              # type of gaussian noise? 'None'/'multi0'/'multi1'/'fake_drop' --> (x+n)/x*n/x*(n+1)/x*s(n) 
            # TRAINING: COST
            self.cost = 'categorical_crossentropy'                             # cost for T1? 'L2'/'categorical_crossentropy'
            self.cost_T2 = 'crossEntropy' # TODO more                          # cost for T2? 'L2'/'crossEntropy'                       TODO: 'sigmoidal'/'hingeLoss'  
            self.train_T2_gradient_jacobian = False                            # train the T2 params based on gradient jacobian
            self.penalize_T2 = False                                           # apply penalty for T2? 
            self.cost2Type = 'default'                                         # type of T1T2 cost 'default'/'C2-C1' 
            # TRAINING: T2 FD or exact
            self.finiteDiff = False                                            # use finite difference for T2?
            self.FDStyle = '3'                                                 # type of finite difference implementation  '2'/'3'
            self.checkROP = False  # TODO                                      # check ROP operator efficiency                          
            self.T2gradDIY = False  # TODO                                     # use your own ROP operator                              
            self.T2onlySGN = False                                             # consider only the sign for T2 update, not the amount
            # TRAINING: OPTIMIZATION
            self.learnRate1 = 0.001                                            # T1 max step size
            self.learnRate2 = 0.001                                             # T2 max step size
            self.learnFun1 = 'olin'                                             # learning rate schedule for T1? (see LRFunctions for options)
            self.learnFun2 = 'period'                                            # learning rate schedule for T2? 
            self.opt1 = 'adam'                                                 # optimizer for T1? 'adam'/None (None is SGD)
            self.opt2 ='adam'                                                   # optimizer for T2? 'adam'/None (None is SGD)
            self.use_momentum = False                                          # applies both to T1 and T2, set the terms to 0 for either if want to disable for one   
            self.momentum1 = [0.5, 0.9]                                        # T1 max and min momentum values
            self.momentum2 = [0.5, 0.9]                                        # T2 max and min momentum values
            self.momentFun = 'exp'                                             # momentum decay function
            self.halfLife = 1                                                  # decay function parameter, set to be at halfLife*10,000 updates later
            self.triggerT2 = 0                                                 # when to start training with T2  
            self.hessian_T2 = False                                            # apply penalty for T2? 
            self.avC2grad = 'None'                                             # taking averaging of C2grad and how? 'None'/'adam'/'momentum'
            self.MM = 1  # TODO?                                                # for stochastic net: how many parallel samples do we take?  
            # TRAINING: OTHER
            self.batchSize1 = 100
            self.batchSize2 = 100
            self.maxEpoch = 150
            self.seed = 1234
            # TRACKING, PRINTING
            self.trackPerEpoch = 1                                             # how often within epoch track error?
            self.printInterval = 2                                             # how often print error?
            self.printBest = 40000                                             # each updates print best value?
            self.activTrack = ['mean', 'std', 'max',                           # what network statistics are you following?
                               'const', 'spars', 
                               'wmean', 'wstd', 'wmax', 
                               'rnoise', 'rnstd', 
                               'bias', 'a'] 
            self.forVideo = ['a', 'b', 'h']                                    # takes a sample of say 100-200 of those from each layer 
            self.showGrads = False                                             # do you show gradient values?
#            self.listGrads = ['grad', 'grad_rel', 'grad_angle', 'grad_max', 'p_t', 'p_t_rel', 'p_t_angle', 'p_t_max']
            self.listGrads = ['grad', 'grad_angle', 'p_t', 'p_t_angle']        # which gradient measures to track?             
            self.whichGrads = 'all' 
            self.trackGrads = False
            self.trackStat = False
            self.track4Vid = False
            self.track1stFeatures = False  # TODO                            
          
    
    # replace default parameters      
    params = Params()    
    for key, val in replace_params.iteritems():
        assert hasattr(params, key), 'Setting %s does not exist' % key
        setattr(params, key, val)

    # add in case of convolutional network                
    if params.model == 'convnet':        
        params.convLayers = conv_setup(params) 
        params.nLayers = len(params.convLayers)    
      
    if not params.useT2:
        params.train_T2_gradient_jacobian = False
        params.rglrzTrain = []
    
    return params
    
