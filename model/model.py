from configs import *

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model


def name_layer(prefix='Layer', layer_names: list = None) -> str:
    if layer_names is None:
        name = prefix
    else:
        counter = 1
        while True:
            name = f'{prefix}_{counter}'
            if name in layer_names:
                counter += 1
            else:
                layer_names.append(name)
                break
    return name


def circle_mapping(input_data, layer_names: list = None):
    name = name_layer('CircleMapping', layer_names=layer_names)

    def layer(x):
        x = tf.keras.activations.tanh(x)
        x_, y_ = x[..., 0:1], x[..., 1:2]
        x_ *= tf.math.sqrt(1 - tf.math.square(y_) / 2)
        y_ *= tf.math.sqrt(1 - tf.math.square(x_) / 2)
        x = layers.Concatenate(axis=1)([y_, x_])
        return x

    input_layer = layers.Input(shape=(2,))
    output_tensor = layer(input_layer)

    output_layer = Model(input_layer, output_tensor, name=name)
    return output_layer(input_data)


def dense(input_data, nodes: int, activation=layers.ReLU(), bn=True, activation_first=False, dropout=0.0,
          layer_names: list = None):
    name = name_layer('Dense', layer_names=layer_names)

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

    output_layer = Model(input_layer, output_tensor, name=name)
    return output_layer(input_data)


def conv1d_loop(input_data, filters: int, kernel_size=3, strides=1, groups=1, activation=layers.ReLU(),
                bn=True, activation_first=False, dropout=0.0, layer_names: list = None):
    name = name_layer('LoopConv1D', layer_names=layer_names)

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

    output_layer = Model(input_layer, output_tensor, name=name)
    return output_layer(input_data)


def conv1d_trans_loop(input_data, input_loc, filters: int, kernel_size=3, strides=1, groups=1, activation=layers.ReLU(),
                      bn=True, activation_first=False, dropout=0.0, layer_names: list = None):
    name = name_layer('LoopTransConv1D', layer_names=layer_names)

    def layer(x, pos):
        pad_size = int((kernel_size - 1) / 2)
        if pad_size != 0:
            x = layers.Concatenate(axis=1)([x[:, -pad_size:], x, x[:, :pad_size]])
        x = layers.Conv1D(filters, kernel_size, strides, groups=groups)(x)
        x_transform = layers.Dense(filters)(pos)
        x = layers.multiply([x, x_transform])
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
    input_pos = layers.Input(shape=input_loc.shape[1:])
    output_tensor = layer(input_layer, input_pos)

    output_layer = Model((input_layer, input_pos), output_tensor, name=name)
    return output_layer((input_data, input_loc))


def pool1d_loop(input_data, pool_type='max', pool_size=3, strides=2, layer_names: list = None):
    name = name_layer(f'{pool_type}Pool', layer_names=layer_names)

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

    output_layer = Model(input_layer, output_tensor, name=name)
    return output_layer(input_data)


# Slightly better, but computationally heavy methods
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


# Slightly better, but computationally heavy methods
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
    def model(input_data, input_loc):
        layer_names = []
        x = input_data
        x1 = input_loc

        x = conv1d_trans_loop(x, x1, 16, kernel_size=11, strides=2, bn=True, dropout=0.2, layer_names=layer_names)
        x = conv1d_trans_loop(x, x1, 32,  kernel_size=3, strides=2, bn=True, dropout=0.2, layer_names=layer_names)
        x = conv1d_trans_loop(x, x1, 64,  kernel_size=3, strides=2, bn=True, dropout=0.2, layer_names=layer_names)
        x = conv1d_trans_loop(x, x1, 128, kernel_size=3, strides=2, bn=True, dropout=0.2, layer_names=layer_names)
        x = conv1d_trans_loop(x, x1, 256, kernel_size=3, strides=2, bn=True, dropout=0.2, layer_names=layer_names)

        x = layers.Flatten()(x)
        x = dense(x, 256, bn=True, dropout=0.1, layer_names=layer_names)
        x = dense(x, 128, dropout=0.1, layer_names=layer_names)
        x = dense(x, 64, layer_names=layer_names)
        x = layers.Concatenate(axis=1)([x, x1])

        x = layers.Dense(2)(x)
        x = circle_mapping(x, layer_names=layer_names)
        return x

    np.random.seed(MODEL_SEED)
    tf.random.set_seed(MODEL_SEED)
    input_layer = layers.Input(shape=(MODEL_VISION, 2))
    input_pos = layers.Input(shape=(2,))
    output_tensors = model(input_layer, input_pos)
    model_output = Model((input_layer, input_pos), output_tensors, name='loop_net')
    return model_output


if __name__ == '__main__':
    loop_net = create_model()
    loop_net.summary()
    print()

    in_ = (tf.random.uniform(shape=(16, MODEL_VISION, 2)),
           tf.random.uniform(shape=(16, 2)))
    out = loop_net(in_, training=True)
    print(out)

    # a = loop_net.save_weights('/home/shin/Desktop/TouhouBulletHell/checkpoint/cp-01')
    # print(a)
    from tensorflow.keras.utils import plot_model
    plot_model(loop_net, 'file.png', show_shapes=True)
