#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from keras.engine import Layer, InputSpec
from keras import activations, initializations, regularizers, constraints
from keras import backend as K
from keras.layers.convolutional import conv_output_length
import numpy as np

class HighwayConv2D(Layer):
    '''
    Convolutional HighwayLayer

    # Arguments
        nb_filter: Number of convolution filters to use.
        nb_row: Number of rows in the convolution kernel.
        nb_col: Number of columns in the convolution kernel.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)), or alternatively,
            Theano function to use for weights initialization.
            This parameter is only relevant if you don't pass
            a `weights` argument.
        activation: name of activation function to use for the transform
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list of numpy arrays to set as initial weights.
        border_mode: 'valid' or 'same'.
        subsample: tuple of length 2. Factor by which to subsample output.
            Also called strides elsewhere.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        b_constraint: instance of the [constraints](../constraints.md) module,
            applied to the bias.
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode is it at index 3.
            It defaults to the `image_dim_ordering` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "th".
        bias: whether to include a bias (i.e. make the layer affine rather than linear).

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.

    # Output shape
        4D tensor with shape:
        `(samples, nb_filter, new_rows, new_cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, nb_filter)` if dim_ordering='tf'.
        `rows` and `cols` values might have changed due to padding.
    '''
    def __init__(self, nb_filter, nb_row, nb_col,
                 init='glorot_uniform', bT_init=-2, activation='linear',
                 weights=None, border_mode='valid', subsample=(1, 1),
                 dim_ordering=K.image_dim_ordering(),
                 WT_regularizer=None, bT_regularizer=None,
                 WH_regularizer=None, bH_regularizer=None,
                 activity_regularizer=None,
                 WT_constraint=None, bT_constraint=None,
                 WH_constraint=None, bH_constraint=None,
                 T_bias=True, H_bias=True, **kwargs):

        if border_mode not in {'valid', 'same'}:
            raise Exception('Invalid border mode for Convolution2D:', border_mode)
        self.nb_filter = nb_filter
        self.nb_row = nb_row
        self.nb_col = nb_col
        self.init = initializations.get(init, dim_ordering=dim_ordering)
        self.activation = activations.get(activation)
        assert border_mode in {'valid', 'same'}, 'border_mode must be in {valid, same}'
        self.border_mode = border_mode
        self.subsample = tuple(subsample)
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering

        self.WT_regularizer = regularizers.get(WT_regularizer)
        self.bT_regularizer = regularizers.get(bT_regularizer)

        self.WH_regularizer = regularizers.get(WH_regularizer)
        self.bH_regularizer = regularizers.get(bH_regularizer)

        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.WT_constraint = constraints.get(WT_constraint)
        self.bT_constraint = constraints.get(bT_constraint)

        self.WH_constraint = constraints.get(WH_constraint)
        self.bH_constraint = constraints.get(bH_constraint)

        self.T_bias = T_bias
        self.H_bias = H_bias
        self.input_spec = [InputSpec(ndim=4)]
        self.initial_weights = weights
        self.bT_init = bT_init

        super(HighwayConv2D, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.dim_ordering == 'th':
            stack_size = input_shape[1]
            self.W_shape = (self.nb_filter, stack_size, self.nb_row, self.nb_col)
        elif self.dim_ordering == 'tf':
            stack_size = input_shape[3]
            self.W_shape = (self.nb_row, self.nb_col, stack_size, self.nb_filter)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        self.WT = self.init(self.W_shape, name='{}_WT'.format(self.name))
        self.WH = self.init(self.W_shape, name='{}_WH'.format(self.name))
        self.trainable_weights = [self.WT, self.WH]

        if self.T_bias:
            self.bT = K.zeros((self.nb_filter,), name='{}_bT'.format(self.name))
            K.set_value(self.bT, np.ones((self.nb_filter,), dtype=np.float32) * self.bT_init)
            self.trainable_weights.append(self.bT)
        if self.H_bias:
            self.bH = K.zeros((self.nb_filter,), name='{}_bH'.format(self.name))
            self.trainable_weights.append(self.bH)

        self.regularizers = []

        if self.WT_regularizer:
            self.WT_regularizer.set_param(self.WT)
            self.regularizers.append(self.WT_regularizer)

        if self.WH_regularizer:
            self.WH_regularizer.set_param(self.WH)
            self.regularizers.append(self.WH_regularizer)

        if self.T_bias and self.bT_regularizer:
            self.bT_regularizer.set_param(self.bT)
            self.regularizers.append(self.bT_regularizer)

        if self.H_bias and self.bH_regularizer:
            self.bH_regularizer.set_param(self.bH)
            self.regularizers.append(self.bH_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        self.constraints = {}
        if self.WT_constraint:
            self.constraints[self.WT] = self.WT_constraint
        if self.WH_constraint:
            self.constraints[self.WH] = self.WH_constraint

        if self.T_bias and self.bT_constraint:
            self.constraints[self.bT] = self.bT_constraint
        if self.H_bias and self.bH_constraint:
            self.constraints[self.bH] = self.bH_constraint

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def get_output_shape_for(self, input_shape):
        if self.dim_ordering == 'th':
            rows = input_shape[2]
            cols = input_shape[3]
        elif self.dim_ordering == 'tf':
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        rows = conv_output_length(rows, self.nb_row,
                                  self.border_mode, self.subsample[0])
        cols = conv_output_length(cols, self.nb_col,
                                  self.border_mode, self.subsample[1])

        if self.dim_ordering == 'th':
            return (input_shape[0], self.nb_filter, rows, cols)
        elif self.dim_ordering == 'tf':
            return (input_shape[0], rows, cols, self.nb_filter)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def call(self, x, mask=None):
        output_H = K.conv2d(x, self.WH, strides=self.subsample,
                            border_mode=self.border_mode,
                            dim_ordering=self.dim_ordering,
                            filter_shape=self.W_shape)
        if self.H_bias:
            if self.dim_ordering == 'th':
                output_H += K.reshape(self.bH, (1, self.nb_filter, 1, 1))
            elif self.dim_ordering == 'tf':
                output_H += K.reshape(self.bH, (1, 1, 1, self.nb_filter))
            else:
                raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        output_H = self.activation(output_H)

        output_T = K.conv2d(x, self.WT, strides=self.subsample,
                            border_mode=self.border_mode,
                            dim_ordering=self.dim_ordering,
                            filter_shape=self.W_shape)

        if self.T_bias:
            if self.dim_ordering == 'th':
                output_T += K.reshape(self.bT, (1, self.nb_filter, 1, 1))
            elif self.dim_ordering == 'tf':
                output_T += K.reshape(self.bT, (1, 1, 1, self.nb_filter))
            else:
                raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        output_T = K.sigmoid(output_T)

        output = output_T * output_H + (1. - output_T) * x

        return output

    def get_config(self):
        config = {'nb_filter': self.nb_filter,
                  'nb_row': self.nb_row,
                  'nb_col': self.nb_col,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'border_mode': self.border_mode,
                  'subsample': self.subsample,
                  'dim_ordering': self.dim_ordering,
                  'WT_regularizer': self.WT_regularizer.get_config() if self.WT_regularizer else None,
                  'bT_regularizer': self.bT_regularizer.get_config() if self.bT_regularizer else None,
                  'WH_regularizer': self.WH_regularizer.get_config() if self.WH_regularizer else None,
                  'bH_regularizer': self.bH_regularizer.get_config() if self.bH_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'WT_constraint': self.WT_constraint.get_config() if self.WT_constraint else None,
                  'bT_constraint': self.bT_constraint.get_config() if self.bT_constraint else None,
                  'W_constraint': self.WH_constraint.get_config() if self.WH_constraint else None,
                  'b_constraint': self.bH_constraint.get_config() if self.bH_constraint else None,
                  'T_bias': self.T_bias,
                  'H_bias': self.H_bias}
        base_config = super(HighwayConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
