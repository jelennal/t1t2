#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from keras.models import Sequential
from keras.layers import Activation, Flatten, Dropout
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianNoise
from keras.optimizers import Adam
from keras.utils import np_utils

# TODO: Replace with preprocessed version
from keras.datasets import cifar10

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)


def add_bn_noise(model):
    model.add(BatchNormalization(mode=1))  # TODO: fix gamma
    model.add(GaussianNoise(sigma=0.3))

l = 0.001

model = Sequential()

model.add(Dropout(0.2, input_shape=(3, 32, 32)))
model.add(Convolution2D(96, 3, 3, border_mode='same', W_regularizer=l2(l)))
model.add(Activation('relu'))
model.add(Convolution2D(96, 3, 3, border_mode='same', W_regularizer=l2(l)))
model.add(Activation('relu'))
model.add(Convolution2D(96, 3, 3, border_mode='valid', subsample=(2, 2), W_regularizer=l2(l)))
model.add(Dropout(0.5))

model.add(Convolution2D(192, 3, 3, border_mode='same', W_regularizer=l2(l)))
model.add(Activation('relu'))
model.add(Convolution2D(192, 3, 3, border_mode='same', W_regularizer=l2(l)))
model.add(Activation('relu'))
model.add(Convolution2D(192, 3, 3, border_mode='valid', subsample=(2, 2), W_regularizer=l2(l)))
model.add(Dropout(0.5))

model.add(Convolution2D(192, 3, 3, border_mode='same', W_regularizer=l2(l)))
model.add(Activation('relu'))
model.add(Convolution2D(192, 1, 1, border_mode='valid', W_regularizer=l2(l)))
model.add(Activation('relu'))
model.add(Convolution2D(10, 1, 1, border_mode='valid', W_regularizer=l2(l)))

model.add(AveragePooling2D(pool_size=(6, 6)))
model.add(Flatten())
model.add(Activation('softmax'))


adam = Adam()

model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

batch_size = 128
epochs = 100

model.fit(X_train, Y_train,
          batch_size=128,
          nb_epoch=100,
          validation_data=(X_test, Y_test),
          shuffle=True)