# T1-T2 Hyperparameter Tuning

Theano implementation of T1-T2 method for tuning continuous hyperparameters.
Paper: [http://arxiv.org/abs/1511.06727](http://arxiv.org/abs/1511.06727)

![click me][traject]

Currently supporting:
- architectures: mlp, cnn
- datasets: [mnist](http://yann.lecun.com/exdb/mnist/), [svhn](http://ufldl.stanford.edu/housenumbers/), [cifar-10](https://www.cs.toronto.edu/~kriz/cifar.html) and [not_mnist](http://yaroslavvb.blogspot.fi/2011/09/notmnist-dataset.html) data sets
- regularization: batch normalization; various versions of additive and multiplicative gaussian noise; 
L1, L2, Lmax, soft Lmax penalties; drop-out, per-batch drop-out
- training regularizers: all gaussian noise, L2, soft Lmax; 
parametrized per unit, per map (for cnn), per layer, per network
- optimizers: SGD, momentum, adam
- T2 gradients: via L-op, via finite difference
- monitoring: various network activation and parameter statistics, gradient norms and angles 

This version was implemented partially as an exercise, more efficient implementation will be developed in [keras](https://github.com/fchollet/keras/).


Test performance with initial random hyperparameters vs. same model just tuned hyperparameter:
- [MNIST](https://github.com/jelennal/t1t2/blob/master/pics/beforeafter_in_scale(mnist).png) 
- [SVHN](https://github.com/jelennal/t1t2/blob/master/pics/beforeafter_1(svhn).png) 

(different symbols correspond to different experiment setups: varying network architecture, number and degree of freedom of hyperparameters; x-axis: test error before tuning, y-axis: test error after tuning)

[traject]: https://github.com/jelennal/t1t2/blob/master/pics/trajectories%20in%20hyperspace%20(mnist).png "Hyperparameter values during training with T1-T2, illustrated in hyperparameter space."
