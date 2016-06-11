#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from keras.models import Sequential
from keras.layers import Activation, Flatten, Dropout
from keras.layers import Convolution2D, AveragePooling2D
from keras.regularizers import l2
from keras.callbacks import LearningRateScheduler, Callback, BaseLogger
from keras.optimizers import Adam, SGD
from keras.utils import np_utils
from keras.datasets import cifar10
from sacred import Experiment
import numpy as np

ex = Experiment('keras_allconv_cifar10')


@ex.config
def cfg():
    act_func = 'relu'
    learning_rate = 0.25
    schedule = [0, 200, 250, 300]
    max_epochs = 350
    batch_size = 128
    base_size = 32
    subset = 50000

    momentum = 0.9
    decay = 0.001
    use_adam = False
    verbose = False


@ex.capture
def get_schedule(learning_rate, schedule):
    def sched(epoch_nr):
        return learning_rate * 0.1**(np.searchsorted(schedule, epoch_nr+1)-1)
    return LearningRateScheduler(schedule=sched)


@ex.capture
def prepare_dataset(subset):
    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    X_train = X_train[:subset].astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.
    X_test /= 255.

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train[:subset], 10)
    Y_test = np_utils.to_categorical(y_test, 10)
    return X_train, Y_train, X_test, Y_test


@ex.capture
def build_model(base_size, act_func, decay):
    model = Sequential()

    model.add(Dropout(0.2, input_shape=(3, 32, 32)))
    model.add(Convolution2D(base_size*3, 3, 3, border_mode='same',
                            W_regularizer=l2(decay)))
    model.add(Activation(act_func))
    model.add(Convolution2D(base_size*3, 3, 3, border_mode='same',
                            W_regularizer=l2(decay)))
    model.add(Activation(act_func))
    model.add(Convolution2D(base_size*3, 3, 3, border_mode='valid', subsample=(2, 2),
                            W_regularizer=l2(decay)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(base_size*6, 3, 3, border_mode='same',
                            W_regularizer=l2(decay)))
    model.add(Activation(act_func))
    model.add(Convolution2D(base_size*6, 3, 3, border_mode='same',
                            W_regularizer=l2(decay)))
    model.add(Activation(act_func))
    model.add(Convolution2D(base_size*6, 3, 3, border_mode='valid', subsample=(2, 2),
                            W_regularizer=l2(decay)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(base_size*6, 3, 3, border_mode='same',
                            W_regularizer=l2(decay)))
    model.add(Activation(act_func))
    model.add(Convolution2D(base_size*6, 1, 1, border_mode='valid',
                            W_regularizer=l2(decay)))
    model.add(Activation(act_func))
    model.add(Convolution2D(10, 1, 1, border_mode='valid',
                            W_regularizer=l2(decay)))

    model.add(AveragePooling2D(pool_size=(6, 6)))
    model.add(Flatten())
    model.add(Activation('softmax'))
    return model


@ex.capture
def compile_model(model, learning_rate, momentum, use_adam):
    if use_adam:
        opt = Adam()
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
def run(batch_size, max_epochs, verbose, _run):
    X_train, Y_train, X_test, Y_test = prepare_dataset()
    model = build_model()
    compile_model(model)
    model.fit(X_train, Y_train,
              verbose=verbose,
              batch_size=batch_size,
              nb_epoch=max_epochs,
              validation_data=(X_test, Y_test),
              callbacks=[get_schedule(), BaseLogger(), InfoUpdater(_run)],
              shuffle=True)

    import tempfile
    with tempfile.NamedTemporaryFile(prefix='model_', suffix='.json') as f:
        f.write(model.to_json())
        ex.add_artifact(f.name)

    with tempfile.NamedTemporaryFile(prefix='weights_', suffix='.npz') as f:
        model.save_weights(f.name, overwrite=True)
        ex.add_artifact(f.name)

    return model.evaluate(X_test, Y_test, batch_size, verbose=verbose)
