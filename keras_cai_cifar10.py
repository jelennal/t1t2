#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from keras.models import Sequential
from keras.layers import Activation, Flatten, Dropout
from keras.layers import Convolution2D, AveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.noise import GaussianNoise
from keras.regularizers import l2
from keras.callbacks import LearningRateScheduler, Callback, BaseLogger
from keras.optimizers import Adam, SGD
from keras.utils import np_utils
from keras.datasets import cifar10
from sacred import Experiment
import numpy as np

ex = Experiment('keras_custom_cifar10')


# noinspection PyUnusedLocal
@ex.config
def cfg():
    # Data Preprocessing
    subset = 50000
    preprocessed = 'pylearn2'
    augment = False
    norm_std = True

    # Architecture
    act_func = 'leakyrelu'
    init = 'normal'  # 'he_normal'
    base_size = 32
    leak = 0.1
    std = 0.3
    decay = 0.001
    eps = 1e-6
    bn = "___B___B__"


    # Training
    max_epochs = 350
    batch_size = 128
    schedule = [0, 200, 250, 300]
    momentum = 0.9
    learning_rate = 0.001
    use_adam = True
    verbose = False


@ex.capture
def get_schedule(learning_rate, schedule):
    def sched(epoch_nr):
        return learning_rate * 0.1**(np.searchsorted(schedule, epoch_nr+1)-1)
    return LearningRateScheduler(schedule=sched)


@ex.capture
def prepare_dataset(preprocessed, subset, norm_std):
    if preprocessed is True:
        print('Loading preprocessed dataset')
        ds = np.load('preprocessed_cifar.npz')
        X_train = ds['X_train']
        y_train = ds['Y_train']
        X_t2 = ds['X_t2']
        y_t2 = ds['Y_t2']
        X_train = np.concatenate((X_train, X_t2), axis=0)
        y_train = np.concatenate((y_train, y_t2), axis=0)
        X_test = ds['X_test']
        y_test = ds['Y_test']
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
    elif preprocessed == 'pylearn2':
        print('Loading preprocessed dataset from pylearn2')
        import pickle
        with open('/home/greff/Datasets/pylearn2/cifar10/pylearn2_gcn_whitened/train.pkl', 'rb') as f:
            train_ds = pickle.load(f)
        with open('/home/greff/Datasets/pylearn2/cifar10/pylearn2_gcn_whitened/test.pkl', 'rb') as f:
            test_ds = pickle.load(f)
        X_test = test_ds.get_topological_view().transpose(0, 3, 1, 2)
        y_test = test_ds.y
        X_train = train_ds.get_topological_view().transpose(0, 3, 1, 2)
        y_train = train_ds.y
    else:
        print('Using raw dataset')
        # the data, shuffled and split between train and test sets
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255.
        X_test /= 255.

    if norm_std:
        std = X_train.std()
        X_train /= std
        X_test /= std

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)

    X_train = X_train[:subset]
    Y_train = Y_train[:subset]

    print('X_train shape:', X_train.shape)
    print('Y_train shape:', Y_train.shape)
    print('X_test shape:', X_test.shape)
    print('Y_test shape:', Y_test.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    return X_train, Y_train, X_test, Y_test


@ex.capture
def build_model(base_size, act_func, decay, init, std, eps, leak, bn):
    def add_bn_noise(model, bn="B"):
        if bn == 'B':
            model.add(BatchNormalization(mode=1, epsilon=eps))  # TODO: fix gamma
        model.add(GaussianNoise(sigma=std))

    def add_act(model):
        if act_func == 'leakyrelu':
            model.add(LeakyReLU(alpha=leak))
        else:
            model.add(Activation(act_func))

    model = Sequential()
    # cl1 = cnn_layer('conv',    (3, 3), (3, 96),    (1, 1), 'valid')
    model.add(Convolution2D(base_size*3, 3, 3, border_mode='valid',
                            input_shape=(3, 32, 32), init=init, W_regularizer=l2(decay)))
    add_bn_noise(model, bn[0])
    add_act(model)

    # cl2 = cnn_layer('conv',    (3, 3), (96, 96),   (1, 1), 'full')
    model.add(Convolution2D(base_size*3, 3, 3, border_mode='same', init=init, W_regularizer=l2(decay)))
    add_bn_noise(model, bn[1])
    add_act(model)

    # cl2alt = cnn_layer('conv', (3, 3), (96, 96),   (1, 1), 'full')
    model.add(Convolution2D(base_size*3, 3, 3, border_mode='same', init=init, W_regularizer=l2(decay)))
    add_bn_noise(model, bn[2])
    add_act(model)

    # cl3 = cnn_layer('pool',    (2, 2), (96, 96),   (2, 2), 'dummy', 1, 1)
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    add_bn_noise(model, bn[3])

    # cl4 = cnn_layer('conv',    (3, 3), (96, 192),  (1, 1), 'valid')
    model.add(Convolution2D(base_size*6, 3, 3, border_mode='valid', init=init, W_regularizer=l2(decay)))
    add_bn_noise(model, bn[4])
    add_act(model)

    # cl5 = cnn_layer('conv',    (3, 3), (192, 192), (1, 1), 'full')
    model.add(Convolution2D(base_size*6, 3, 3, border_mode='same', init=init, W_regularizer=l2(decay)))
    add_bn_noise(model, bn[5])
    add_act(model)

    # cl6 = cnn_layer('conv',    (3, 3), (192, 192), (1, 1), 'valid')
    model.add(Convolution2D(base_size*6, 3, 3, border_mode='same', init=init, W_regularizer=l2(decay)))
    add_bn_noise(model, bn[6])
    add_act(model)

    # cl7 = cnn_layer('pool',    (2, 2), (192, 192), (2, 2), 'dummy', 1, 1)
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    add_bn_noise(model, bn[7])

    # cl8  = cnn_layer('conv',   (3, 3), (192, 192), (1, 1), 'valid')
    model.add(Convolution2D(base_size*6, 3, 3, border_mode='same', init=init, W_regularizer=l2(decay)))
    add_bn_noise(model, bn[8])
    add_act(model)

    # cl9  = cnn_layer('conv',   (1, 1), (192, 192), (1, 1), 'valid')
    model.add(Convolution2D(base_size*6, 1, 1, border_mode='valid', init=init, W_regularizer=l2(decay)))
    add_bn_noise(model, bn[9])
    add_act(model)

    # cl10 = cnn_layer('conv',   (1, 1), (192, 10),  (1, 1), 'valid', 0, 0)
    model.add(Convolution2D(10, 1, 1, border_mode='valid', init=init, W_regularizer=l2(decay)))
    add_act(model)

    # cl11 = cnn_layer('average+softmax', (6, 6), (10, 10), (6, 6), 'dummy', 0, 0)
    model.add(AveragePooling2D(pool_size=(6, 6)))
    model.add(Flatten())
    model.add(Activation('softmax'))
    return model


@ex.capture
def compile_model(model, learning_rate, momentum, use_adam):
    if use_adam:
        opt = Adam(learning_rate)
    else:
        opt = SGD(lr=learning_rate, momentum=momentum)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model


class InfoUpdater(Callback):
    def __init__(self, run):
        super(InfoUpdater, self).__init__()
        self.run = run
        self.run.info['logs'] = {}

    def on_epoch_end(self, epoch, logs={}):
        for k, v in logs.items():
            logout = self.run.info['logs'].get(k, [])
            logout.append(v)
            self.run.info['logs'][k] = logout


@ex.automain
def run(batch_size, max_epochs, verbose, augment, _run):
    X_train, Y_train, X_test, Y_test = prepare_dataset()
    model = build_model()
    compile_model(model)

    if not augment:
        print('Not using data augmentation.')
        model.fit(X_train, Y_train,
                  verbose=verbose,
                  batch_size=batch_size,
                  nb_epoch=max_epochs,
                  validation_data=(X_test, Y_test),
                  callbacks=[get_schedule(), BaseLogger(), InfoUpdater(_run)],
                  shuffle=True)
    else:
        print('Using real-time data augmentation.')
        # this will do preprocessing and realtime data augmentation
        datagen = ImageDataGenerator(
            width_shift_range=5./32, # randomly shift images horizontally (fraction of total width)
            height_shift_range=5./32, # randomly shift images vertically (fraction of total height)
            horizontal_flip=True)  # randomly flip images

        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        datagen.fit(X_train)
        # fit the model on the batches generated by datagen.flow()
        model.fit_generator(
            datagen.flow(X_train, Y_train, batch_size=batch_size),
            samples_per_epoch=X_train.shape[0],
            nb_val_samples=X_test.shape[0],
            verbose=verbose,
            nb_epoch=max_epochs,
            validation_data=datagen.flow(X_test, Y_test, batch_size=batch_size),
            callbacks=[get_schedule(), BaseLogger(), InfoUpdater(_run)])

    import tempfile
    with tempfile.NamedTemporaryFile(prefix='model_', suffix='.json') as f:
        f.write(model.to_json())
        ex.add_artifact(f.name)

    with tempfile.NamedTemporaryFile(prefix='weights_', suffix='.npz') as f:
        model.save_weights(f.name, overwrite=True)
        ex.add_artifact(f.name)

    if augment:
        return model.evaluate_generator(datagen.flow(X_test, Y_test, batch_size=batch_size), val_samples=X_test.shape[0], verbose=verbose)
    else:
        return model.evaluate(X_test, Y_test, batch_size, verbose=verbose)

