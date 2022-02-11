from configs import *
from model.model_utils import *

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model


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
