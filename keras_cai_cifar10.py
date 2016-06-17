#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from keras.models import Sequential
from keras.layers import Activation, Flatten, Dropout
from keras.layers import Convolution2D, AveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, ELU
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
    val_subset = 10000
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
    spec = "C3NA C3NA C3NA MBN C6NA C6NA C6NA MBN C6NA c6NA"

    # Training
    max_epochs = 350
    batch_size = 128
    sched_type = 'discreet'
    schedule = [0, 200, 250, 300]
    momentum = 0.9
    learning_rate = 0.001
    use_adam = True
    verbose = False
    tensorboard = False


@ex.named_config
def ladder_baseline():
    spec = "C3BNA C3BNA C3BNA MBN C6BNA C6BNA C6BNA MBN C6BNA c6BNA"
    sched_type = 'linear'
    schedule = [50, 100]
    max_epochs= 100
    use_adam = True
    learning_rate = 0.002
    std = 0.3
    decay = 0.0
    leak = 0.1
    act_func = 'leakyrelu'


@ex.named_config
def allconv_c():
    spec = "C3A C3A mD C6A C6A mD C6A c6A"
    sched_type = 'discreet'
    schedule = [0, 200, 250, 300]
    max_epochs = 350
    use_adam = False
    learning_rate = 0.01
    decay = 0.001
    act_func = 'relu'


@ex.capture
def get_schedule(learning_rate, schedule, sched_type):
    if sched_type == 'discreet':
        def sched(epoch_nr):
            return learning_rate * 0.1**(np.searchsorted(schedule, epoch_nr+1)-1)
    elif sched_type == 'linear':
        assert len(schedule) == 2, schedule

        def sched(epoch_nr):
            if epoch_nr <= schedule[0]:
                return learning_rate
            else:
                return learning_rate * (1.0 - (epoch_nr - schedule[0]) / (schedule[1] - schedule[0]))
    else:
        raise KeyError('Unkown schedule type "{}"'.format(sched_type))
    return LearningRateScheduler(schedule=sched)


@ex.capture
def prepare_dataset(preprocessed, subset, val_subset, norm_std):
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

    idx = np.arange(X_test.shape[0], dtype=np.int)
    np.random.shuffle(idx)
    idx = idx[:val_subset]
    X_test = X_test[idx]
    Y_test = Y_test[idx]
    print('average val-label:', Y_test.argmax(1).mean())

    print('X_train shape:', X_train.shape)
    print('Y_train shape:', Y_train.shape)
    print('X_test shape:', X_test.shape)
    print('Y_test shape:', Y_test.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    return X_train, Y_train, X_test, Y_test


@ex.capture
def build_model_spec(spec, act_func, decay, init, std, eps, leak, base_size):
    model = Sequential()

    def add_act(name):
        if act_func == 'leakyrelu':
            model.add(LeakyReLU(alpha=leak, name=name))
        elif act_func == 'elu':
            model.add(ELU(name=name))
        else:
            model.add(Activation(act_func, name=name))

    size = 3
    for i, block in enumerate(spec.split()):
        kwargs = {} if i else {'input_shape': (3, 32, 32)}
        if block[0] == 'C':
            size = base_size*int(block[1])
            print('===> Conv 3x3 {}'.format(size))
            model.add(Convolution2D(size, 3, 3, border_mode='same', name="conv_{}".format(i),
                                    init=init, W_regularizer=l2(decay), **kwargs))
            rest = block[2:]
        elif block[0] == 'c':
            size = base_size * int(block[1])
            print('===> Conv 1x1 {}'.format(size))
            model.add(Convolution2D(size, 1, 1, init=init, W_regularizer=l2(decay), name="conv_{}".format(i), **kwargs))
            rest = block[2:]
        elif block[0] == 'M':
            print('===> MaxPool 2x2')
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool_{}".format(i)))
            rest = block[1:]
        elif block[0] == 'm':
            print('===> Strided Convolution 3x3 with stride 2x2')
            model.add(Convolution2D(size, 3, 3, init=init, name="convpool_{}".format(i),
                                    W_regularizer=l2(decay), subsample=(2, 2),
                                    border_mode='valid'))
            rest = block[1:]
        else:
            raise KeyError('Unknown type "{}"'.format(block[0]))

        for r in rest:
            if r == 'B':
                print('BatchNorm')
                model.add(BatchNormalization(mode=1, epsilon=eps, name="batchnorm_{}".format(i)))
            elif r == 'N':
                print('GaussianNoise')
                model.add(GaussianNoise(sigma=std, name="noise_{}".format(i)))
            elif r == 'A':
                print('Activation: {}'.format(act_func))
                add_act(name="act_{}".format(i))
            elif r == 'D':
                print('Dropout (p=0.5)')
                model.add(Dropout(0.5, name="dropout_{}".format(i)))
            elif r == 'd':
                model.add(Dropout(0.2, name="dropout02_{}".format(i)))

        print(model.outputs[0]._keras_shape)

    model.add(Convolution2D(10, 1, 1, border_mode='valid', init=init,
                            W_regularizer=l2(decay)))
    add_act(name="act_{}".format(i+1))

    model.add(AveragePooling2D(pool_size=model.outputs[0]._keras_shape[2:]))
    model.add(Flatten())
    model.add(Activation('softmax', name="softmax"))
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
def run(batch_size, max_epochs, verbose, augment, tensorboard, _run):
    X_train, Y_train, X_test, Y_test = prepare_dataset()
    model = build_model_spec()
    compile_model(model)

    callbacks = [get_schedule(), BaseLogger(), InfoUpdater(_run)]

    if tensorboard:
        print('Adding TensorBoard!')

        from keras import backend as K
        class TensorBoard(Callback):
            ''' copied from keras.callbacks'''
            def __init__(self, log_dir='./logs', histogram_freq=0,
                         write_graph=True):
                super(Callback, self).__init__()
                if K._BACKEND != 'tensorflow':
                    raise Exception('TensorBoard callback only works '
                                    'with the TensorFlow backend.')
                self.log_dir = log_dir
                self.histogram_freq = histogram_freq
                self.merged = None
                self.write_graph = write_graph

            def _set_model(self, model):
                import tensorflow as tf
                import keras.backend.tensorflow_backend as KTF

                self.model = model
                self.sess = KTF.get_session()
                if self.histogram_freq and self.merged is None:
                    layers = self.model.layers
                    for layer in layers:
                        if hasattr(layer, 'W'):
                            tf.histogram_summary('{}_W'.format(layer.name),
                                                 layer.W)
                        if hasattr(layer, 'b'):
                            tf.histogram_summary('{}_b'.format(layer.name),
                                                 layer.b)
                        if hasattr(layer, 'output'):
                            tf.histogram_summary('{}_out'.format(layer.name),
                                                 layer.output)
                self.merged = tf.merge_all_summaries()
                if self.write_graph:
                    if tf.__version__ >= '0.8.0':
                        self.writer = tf.train.SummaryWriter(self.log_dir,
                                                             self.sess.graph)
                    else:
                        self.writer = tf.train.SummaryWriter(self.log_dir,
                                                             self.sess.graph_def)
                else:
                    self.writer = tf.train.SummaryWriter(self.log_dir)

            def on_epoch_end(self, epoch, logs={}):
                import tensorflow as tf

                if self.model.validation_data and self.histogram_freq:
                    if epoch % self.histogram_freq == 0:
                        # TODO: implement batched calls to sess.run
                        # (current call will likely go OOM on GPU)
                        if self.model.uses_learning_phase:
                            cut_v_data = len(self.model.inputs)
                            val_data = self.model.validation_data[
                                       :cut_v_data] + [0]
                            tensors = self.model.inputs + [K.learning_phase()]
                        else:
                            val_data = self.model.validation_data
                            tensors = self.model.inputs
                        feed_dict = dict(zip(tensors, val_data))
                        result = self.sess.run([self.merged],
                                               feed_dict=feed_dict)
                        summary_str = result[0]
                        self.writer.add_summary(summary_str, epoch)

                for name, value in logs.items():
                    if name in ['batch', 'size']:
                        continue
                    summary = tf.Summary()
                    summary_value = summary.value.add()
                    summary_value.simple_value = value
                    summary_value.tag = name
                    self.writer.add_summary(summary, epoch)
                self.writer.flush()

        callbacks.append(TensorBoard(log_dir='./logs', histogram_freq=1,
                                     write_graph=True))
    if not augment:
        print('Not using data augmentation.')
        model.fit(X_train, Y_train,
                  verbose=verbose,
                  batch_size=batch_size,
                  nb_epoch=max_epochs,
                  validation_data=(X_test, Y_test),
                  callbacks=callbacks,
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
            callbacks=callbacks)

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

