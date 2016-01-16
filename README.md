# T1-T2

Theano implementation of T1-T2 method for tuning continuous hyperparameters.
Paper: [http://arxiv.org/abs/1511.06727](http://arxiv.org/abs/1511.06727)

Currently supporting:
- architectures: mlp and convolutional
- datasets: mnist and cifar-10 data sets
- regularization: batch normalization; various versions of additive and multiplicative gaussian noise; 
L1, L2, Lmax, soft Lmax penalties; drop-out, per-batch drop-out
- training regularizers: all gaussian noise, L2, soft Lmax; 
parametrized per unit, per map (for convnets), per layer, per network
- optimizers: SGD, momentum, adam
- T2 gradients: via R-op (TODO: implement DIY version, current version does the naive implementation), via finite difference
- monitoring: various network activation and parameter statistics, gradient norms and angles (TODO: fix for convnet) 

This version was implemented as an exercise, more efficient implementation will be developed in [keras](https://github.com/fchollet/keras/).
