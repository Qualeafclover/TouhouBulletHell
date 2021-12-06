try:
    from .model_configs import *
    from .model_utils import *
except ImportError:
    from model_configs import *
    from model_utils import *

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model


def dense(input_data, nodes: int, activation=layers.ReLU(), bn=True, activation_first=False, dropout=0.0,
          layer_names: list = None):
    if layer_names is None:
        layer_names = []

    def layer(x):
        x = layers.Dense(nodes)(x)
        if activation_first and (activation is not None):
            x = activation(x)
        if bn:
            x = layers.BatchNormalization()(x)
        if (not activation_first) and (activation is not None):
            x = activation(x)
        if dropout:
            x = layers.Dropout(dropout)(x)
        return x
    input_layer = layers.Input(shape=input_data.shape[1:])
    output_tensor = layer(input_layer)

    if layer_names is None:
        name = 'Dense'
    else:
        counter = 1
        while True:
            name = f'Dense_{counter}'
            if name in layer_names:
                counter += 1
            else:
                layer_names.append(name)
                break

    output_layer = Model(input_layer, output_tensor, name=name)
    return output_layer(input_data)


def conv1d_loop(input_data, filters: int, kernel_size=3, strides=1, groups=1, activation=layers.ReLU(),
                bn=True, activation_first=False, dropout=0.0, layer_names: list = None):
    if layer_names is None:
        layer_names = []

    def layer(x):
        pad_size = int((kernel_size - 1) / 2)
        if pad_size != 0:
            x = layers.Concatenate(axis=1)([x[:, -pad_size:], x, x[:, :pad_size]])
        x = layers.Conv1D(filters, kernel_size, strides, groups=groups)(x)
        if activation_first and (activation is not None):
            x = activation(x)
        if bn:
            x = layers.BatchNormalization()(x)
        if (not activation_first) and (activation is not None):
            x = activation(x)
        if dropout:
            x = layers.Dropout(dropout)(x)
        return x

    input_layer = layers.Input(shape=input_data.shape[1:])
    output_tensor = layer(input_layer)

    if layer_names is None:
        name = 'LoopConv1D'
    else:
        counter = 1
        while True:
            name = f'LoopConv1D_{counter}'
            if name in layer_names:
                counter += 1
            else:
                layer_names.append(name)
                break

    output_layer = Model(input_layer, output_tensor, name=name)
    return output_layer(input_data)


def pool1d_loop(input_data, pool_type='max', pool_size=3, strides=2, layer_names: list = None):
    if layer_names is None:
        layer_names = []

    def layer(x):
        pad_size = int((pool_size - 1) / 2)
        if pad_size != 0:
            x = layers.Concatenate(axis=1)([x[:, -pad_size:], x, x[:, :pad_size]])
        if pool_type == 'max':
            x = layers.MaxPooling1D(pool_size=pool_size, strides=strides)(x)
        elif pool_type == 'avg':
            x = layers.AveragePooling1D(pool_size=pool_size, strides=strides)(x)
        else:
            raise NotImplementedError
        return x

    input_layer = layers.Input(shape=input_data.shape[1:])
    output_tensor = layer(input_layer)

    if layer_names is None:
        name = f'{pool_type}Pool'
    else:
        counter = 1
        while True:
            name = f'{pool_type}Pool_{counter}'
            if name in layer_names:
                counter += 1
            else:
                layer_names.append(name)
                break

    output_layer = Model(input_layer, output_tensor, name=name)
    return output_layer(input_data)


def mapping_network(input_data, nodes: int, depth: int, activation=layers.ReLU(),
                    bn=True, activation_first=False, layer_names: list = None):
    def layer(x):
        for n_depth in range(depth):
            x = dense(x, nodes=nodes, activation=activation, bn=bn,
                      activation_first=activation_first, dropout=0.0, layer_names=layer_names)
        return x
    input_layer = layers.Input(shape=input_data.shape[1:])
    output_tensor = layer(input_layer)

    if layer_names is None:
        name = 'Mapping'
    else:
        counter = 1
        while True:
            name = f'Mapping_{counter}'
            if name in layer_names:
                counter += 1
            else:
                layer_names.append(name)
                break

    output_layer = Model(input_layer, output_tensor, name=name)
    return output_layer(input_data)


# Outdated method
def conv1d_loop_resnet_block(input_data, input_filters: int, output_filters: int,
                             kernel_size=3, groups=1, activation=layers.ReLU(), down_sample=False):
    x = input_data
    x = conv1d_loop(x, filters=input_filters, kernel_size=1, groups=groups, activation=activation)
    x = conv1d_loop(x, filters=input_filters, kernel_size=kernel_size, groups=groups, activation=activation)
    x = conv1d_loop(x, filters=output_filters, kernel_size=1, groups=groups)
    x = x + conv1d_loop(input_data, filters=output_filters, kernel_size=1, groups=groups)
    x = activation(x)
    if down_sample:
        x = pool1d_loop(x, pool_size=3, strides=2)
    return x


def conv1d_loop_inception_block(input_data, inception_filters: list, bn=True, activation=layers.ReLU(), dropout=0.0,
                                layer_names: list = None):
    if layer_names is None:
        layer_names = []
    assert len(inception_filters) >= 2
    xs = []
    x = input_data
    for depth in range(len(inception_filters)):
        filters = inception_filters.pop(0)
        if depth == 0:
            x_ = pool1d_loop(x, pool_type='avg', pool_size=3, strides=1, layer_names=layer_names)
            x_ = conv1d_loop(x_, filters=filters, kernel_size=1, strides=1,
                             bn=bn, activation=activation, dropout=dropout, layer_names=layer_names)
        else:
            x_ = conv1d_loop(x, filters=filters, kernel_size=1, strides=1,
                             bn=bn, activation=activation, layer_names=layer_names)
            for _ in range(depth-1):
                x_ = conv1d_loop(x_, filters=filters, kernel_size=3, strides=1,
                                 bn=bn, activation=activation, layer_names=layer_names)
        xs.append(x_)
    x = layers.Concatenate(axis=2)(xs)
    return x


def create_model() -> Model:
    # def model(input_data, input_loc):
    #     layer_names = []
    #     x = input_data
    #
    #     x = conv1d_loop(x, filters=32, kernel_size=11, strides=2, bn=True, layer_names=layer_names)
    #     x = conv1d_loop(x, filters=32, kernel_size=3, bn=True, layer_names=layer_names)
    #     x = conv1d_loop(x, filters=64, kernel_size=3, bn=True, layer_names=layer_names)
    #     x = pool1d_loop(x, layer_names=layer_names)
    #
    #     x = conv1d_loop(x, filters=80, kernel_size=3, bn=True, layer_names=layer_names)
    #     x = conv1d_loop(x, filters=192, kernel_size=3, strides=2, bn=True, layer_names=layer_names)
    #     x = conv1d_loop(x, filters=288, kernel_size=3, bn=True, layer_names=layer_names)
    #
    #     x = conv1d_loop_inception_block(x, inception_filters=[32, 64, 128, 32], bn=True, layer_names=layer_names)
    #     x = conv1d_loop_inception_block(x, inception_filters=[64, 128, 192, 96], bn=True, layer_names=layer_names)
    #     x = pool1d_loop(x, layer_names=layer_names)
    #
    #     x = conv1d_loop_inception_block(x, inception_filters=[64, 128, 256, 64], bn=True, layer_names=layer_names)
    #     x = conv1d_loop_inception_block(x, inception_filters=[128, 256, 320, 128], bn=True, layer_names=layer_names)
    #     x_ = layers.AveragePooling1D(pool_size=16)(x)
    #     x_ = layers.Flatten()(x_)
    #     x = pool1d_loop(x, layer_names=layer_names)
    #
    #     x = conv1d_loop_inception_block(x, inception_filters=[128, 256, 320, 128], bn=True, layer_names=layer_names)
    #     x = conv1d_loop_inception_block(x, inception_filters=[128, 384, 384, 128], bn=True, layer_names=layer_names)
    #     x = layers.AveragePooling1D(pool_size=8)(x)
    #
    #     x = layers.Flatten()(x)
    #     x = layers.Concatenate(axis=-1)([x, x_])
    #     x = layers.Dropout(rate=0.3)(x)
    #     x = dense(x, 128, bn=True, dropout=0.1, layer_names=layer_names)
    #     x = dense(x, 64, layer_names=layer_names)
    #     x = layers.Concatenate(axis=1)([x, input_loc])
    #     x = layers.Dense(MODEL_OUTPUT, activation='sigmoid')(x)
    #     return x
    def model(input_data, input_loc):
        layer_names = []
        x = input_data

        x = conv1d_loop(x, filters=16, kernel_size=11, strides=2, bn=True, dropout=0.2, layer_names=layer_names)
        x = conv1d_loop(x, filters=32,  kernel_size=3, strides=2, bn=True, dropout=0.2, layer_names=layer_names)
        x = conv1d_loop(x, filters=64,  kernel_size=3, strides=2, bn=True, dropout=0.2, layer_names=layer_names)
        x = conv1d_loop(x, filters=128, kernel_size=3, strides=2, bn=True, dropout=0.2, layer_names=layer_names)
        x = conv1d_loop(x, filters=256, kernel_size=3, strides=2, bn=True, dropout=0.2, layer_names=layer_names)

        x = layers.Flatten()(x)
        x = dense(x, 256, bn=True, dropout=0.1, layer_names=layer_names)
        x = dense(x, 128, dropout=0.1, layer_names=layer_names)
        x = dense(x, 64, layer_names=layer_names)
        x = layers.Concatenate(axis=1)([x, input_loc])
        x = layers.Dense(MODEL_OUTPUT, activation='sigmoid')(x)
        return x

    np.random.seed(MODEL_SEED)
    tf.random.set_seed(MODEL_SEED)
    input_layer = layers.Input(shape=(MODEL_VISION, 2))
    input_pos = layers.Input(shape=(2,))
    output_tensors = model(input_layer, input_pos)
    model_output = Model((input_layer, input_pos), output_tensors, name='loop_net')
    return model_output


def create_encoder() -> tf.keras.Model:
    def model(input_data):
        x = input_data
        groups = x.shape[-1]

        x = conv1d_loop(x, filters=32, kernel_size=11, strides=2, groups=groups)
        x = conv1d_loop(x, filters=64, kernel_size=3, strides=2, groups=groups)
        x = conv1d_loop(x, filters=128, kernel_size=3, strides=2, groups=groups)

        x = layers.Flatten()(x)
        return x

    np.random.seed(MODEL_SEED)
    tf.random.set_seed(MODEL_SEED)
    input_layer = layers.Input(shape=(MODEL_VISION, 2))
    output_tensors = model(input_layer)
    model_output = tf.keras.Model(input_layer, output_tensors)
    return model_output


def create_fc_encoder() -> tf.keras.Model:
    def model(encoded, position, test_val):
        x = layers.Concatenate(axis=1)([encoded, position, test_val])
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dense(1)(x)

        return x

    np.random.seed(MODEL_SEED)
    tf.random.set_seed(MODEL_SEED)
    encoded_in = layers.Input(shape=(32 * 128,))  # 4098
    position_in = layers.Input(shape=(2,))
    test_val_in = layers.Input(shape=(5,))
    output_tensors = model(encoded_in, position_in, test_val_in)
    model_output = tf.keras.Model((encoded_in, position_in, test_val_in), output_tensors)
    return model_output


if __name__ == '__main__':
    loop_net = create_model()
    loop_net.summary()
    print()

    # disc_net = create_encoder()
    # disc_net.summary()

    # in_ = (tf.random.uniform(shape=(8, MODEL_VISION, 2)),
    #        tf.random.uniform(shape=(8, 2)))
    # out = loop_net(in_)
    # print(out)

    # a = loop_net.save_weights('/home/shin/Desktop/TouhouBulletHell/checkpoint/cp-01')
    # print(a)
    from tensorflow.keras.utils import plot_model
    plot_model(loop_net, 'file.png', show_shapes=True)
