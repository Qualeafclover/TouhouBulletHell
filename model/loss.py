import numpy as np
import tensorflow as tf
from tensorflow.keras import losses


class MovementError(losses.Loss):
    def call(self, y_true, y_predict):
        y_true = tf.cast(y_true, y_predict.dtype)
        distance = (y_true[..., 2]**2 + y_predict[..., 2]**2 - 2*y_true[..., 2]*y_predict[..., 2]*tf.math.cos(
            tf.math.atan2(y_true[..., 1], y_true[..., 0]) - tf.math.atan2(y_predict[..., 1], y_predict[..., 0])
        )) ** 0.5
        return tf.reduce_mean(distance, axis=-1)


if __name__ == '__main__':
    true = np.array([[0.707, 0.707, 1.00],
                     [-1.00, -0.00, 0.00]], dtype=np.float32)

    pred = tf.constant([[0.0, -1.0, 1.0],
                        [0.0, 1.0, 0.0]], dtype=tf.float32)

    me = MovementError()
    print(me(true, pred))
