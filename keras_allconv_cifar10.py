#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from keras.models import Sequential
from keras.layers import Activation, Flatten, Dropout
from keras.layers import Convolution2D, AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.callbacks import LearningRateScheduler, Callback, BaseLogger
from keras.optimizers import Adam, SGD
from keras.utils import np_utils
from keras.datasets import cifar10
from sacred import Experiment
import numpy as np

ex = Experiment('keras_allconv_cifar10')


# noinspection PyUnusedLocal
@ex.config
def cfg():
    act_func = 'relu'
    learning_rate = 0.25
    schedule = [0, 200, 250, 300]
    max_epochs = 350
    batch_size = 128
    base_size = 32
    subset = 50000
    preprocessed = True
    augment = True

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
def prepare_dataset(preprocessed, subset):
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

