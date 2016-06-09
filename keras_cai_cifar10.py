#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from keras.models import Sequential
from keras.layers import Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
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


model = Sequential()

# cl1 = cnn_layer('conv',    (3, 3), (3, 96),    (1, 1), 'valid')
model.add(Convolution2D(96, 3, 3, border_mode='valid', input_shape=(3, 32, 32)))
add_bn_noise(model)
model.add(LeakyReLU(alpha=0.1))

# cl2 = cnn_layer('conv',    (3, 3), (96, 96),   (1, 1), 'full')
model.add(Convolution2D(96, 3, 3, border_mode='same'))
add_bn_noise(model)
model.add(LeakyReLU(alpha=0.1))

# cl2alt = cnn_layer('conv', (3, 3), (96, 96),   (1, 1), 'full')
model.add(Convolution2D(96, 3, 3, border_mode='same'))
add_bn_noise(model)
model.add(LeakyReLU(alpha=0.1))

# cl3 = cnn_layer('pool',    (2, 2), (96, 96),   (2, 2), 'dummy', 1, 1)
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
add_bn_noise(model)

# cl4 = cnn_layer('conv',    (3, 3), (96, 192),  (1, 1), 'valid')
model.add(Convolution2D(192, 3, 3, border_mode='valid'))
add_bn_noise(model)
model.add(LeakyReLU(alpha=0.1))

# cl5 = cnn_layer('conv',    (3, 3), (192, 192), (1, 1), 'full')
model.add(Convolution2D(192, 3, 3, border_mode='same'))
add_bn_noise(model)
model.add(LeakyReLU(alpha=0.1))

# cl6 = cnn_layer('conv',    (3, 3), (192, 192), (1, 1), 'valid')
model.add(Convolution2D(192, 3, 3, border_mode='same'))
add_bn_noise(model)
model.add(LeakyReLU(alpha=0.1))

# cl7 = cnn_layer('pool',    (2, 2), (192, 192), (2, 2), 'dummy', 1, 1)
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
add_bn_noise(model)

# cl8  = cnn_layer('conv',   (3, 3), (192, 192), (1, 1), 'valid')
model.add(Convolution2D(192, 3, 3, border_mode='same'))
add_bn_noise(model)
model.add(LeakyReLU(alpha=0.1))

# cl9  = cnn_layer('conv',   (1, 1), (192, 192), (1, 1), 'valid')
model.add(Convolution2D(192, 1, 1, border_mode='valid'))
add_bn_noise(model)
model.add(LeakyReLU(alpha=0.1))

# cl10 = cnn_layer('conv',   (1, 1), (192, 10),  (1, 1), 'valid', 0, 0)
model.add(Convolution2D(10,  1, 1, border_mode='valid'))
model.add(LeakyReLU(alpha=0.1))

# cl11 = cnn_layer('average+softmax', (6, 6), (10, 10), (6, 6), 'dummy', 0, 0)
model.add(AveragePooling2D(pool_size=(6, 6)))
model.add(Flatten())
model.add(Activation('softmax'))

adam = Adam(lr=0.002)

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