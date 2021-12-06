from model import create_model
from images import DataLoader
from logger import Logger, Timer
from configs import *
from logger.tensorboard_log import TensorboardLogger
from logger.checkpoint import Checkpoint

import numpy as np
import tensorflow as tf
from tensorflow.keras import losses, optimizers, metrics

import datetime

logger = Logger()

dl = DataLoader()
timer = Timer((len(dl.train_ds) + len(dl.test_ds)) * TRAIN_EPOCHS)
tbl = TensorboardLogger(writer_names=('train', 'test', 'epoch_avg'))
cp = Checkpoint()

if tbl.url is not None:
    logger.log(f'Tensorboard started on {tbl.url}')

model = create_model()
loss_object = losses.MeanSquaredError()
optimizer = optimizers.Adam(learning_rate=0.001)
metrics_loss = metrics.Mean(name='loss')
metrics_accuracy = metrics.BinaryAccuracy(name='accuracy')

logger.log('Model initiation complete.')
logger.log('Starting training.')

step = 0
timer.start()
for epoch_num in range(1, TRAIN_EPOCHS+1):
    metrics_loss.reset_states()
    metrics_accuracy.reset_states()
    for data in dl.train_ds:  # 'player', 'bullet', 'ctrl', 'player_loc', 'vision'
        x_1, x_2 = data['vision'][..., 1:3], data['player_loc']
        y = data['ctrl']

        with tf.GradientTape() as tape:
            prediction = model((x_1, x_2), training=True)
            loss = loss_object(y, prediction)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        metrics_loss(loss)
        metrics_accuracy(y, prediction)

        step += 1
        timer.step()
        logger.log(
            phase='train',
            epoch=f'{epoch_num} / {TRAIN_EPOCHS}',
            batch=f'{dl.train_ds.batch_count} / {len(dl.train_ds)}',
            time_spent=datetime.timedelta(seconds=round(timer.spent())),
            est_time_left=datetime.timedelta(seconds=round(timer.left())),
            avg_loss=f'{metrics_loss.result():.3f}',
            avg_accuracy=f'{metrics_accuracy.result():.3f}',
            loss=f'{loss.numpy():.3f}',
            accuracy=f'{np.mean(metrics.binary_accuracy(y, prediction)):.3f}',
            sample_x=prediction.numpy()[0].round(3),
            sample_y=y[0].round(3),
        )
        tbl.write_summary(
            loss=float(loss.numpy()),
            accuracy=float(np.mean(metrics.binary_accuracy(y, prediction))),
            writer='train',
            step=step,
        )
        tbl.write_summary(
            loss=float(metrics_loss.result()),
            accuracy=float(metrics_accuracy.result()),
            writer='epoch_avg',
            step=step,
        )

    metrics_loss.reset_states()
    metrics_accuracy.reset_states()
    for data in dl.test_ds:   # 'player', 'bullet', 'ctrl', 'player_loc', 'vision'
        x_1, x_2 = data['vision'][..., 1:3], data['player_loc']
        y = data['ctrl']

        prediction = model((x_1, x_2), training=False)
        loss = loss_object(y, prediction)
        metrics_loss(loss)
        metrics_accuracy(y, prediction)

        timer.step()
        logger.log(
            phase='test',
            epoch=f'{epoch_num} / {TRAIN_EPOCHS}',
            batch=f'{dl.test_ds.batch_count} / {len(dl.test_ds)}',
            time_spent=datetime.timedelta(seconds=round(timer.spent())),
            est_time_left=datetime.timedelta(seconds=round(timer.left())),
            avg_loss=f'{metrics_loss.result():.3f}',
            avg_accuracy=f'{metrics_accuracy.result():.3f}',
            loss=f'{loss.numpy():.3f}',
            accuracy=f'{np.mean(metrics.binary_accuracy(y, prediction)):.3f}',
            sample_x=prediction.numpy()[0].round(3),
            sample_y=y[0].round(3),
        )
    tbl.write_summary(
        loss=metrics_loss.result(),
        accuracy=float(metrics_accuracy.result()),
        writer='test',
        step=step,
    )

    logger.log(f'END OF EPOCH {epoch_num}',
               time_spent=datetime.timedelta(seconds=round(timer.spent())),
               est_time_left=datetime.timedelta(seconds=round(timer.left())),
               test_loss=f'{metrics_loss.result():.3f}',
               test_accuracy=f'{metrics_accuracy.result():.3f}'
               )
    save_dir = cp.save(model=model, loss=metrics_loss.result(), epoch_num=epoch_num)
    if save_dir is None:
        logger.log('Model not saved.',
                   loss=f'{metrics_loss.result():.3f}',
                   best_loss=f'{cp.best_loss:.3f}',
                   logtype='W')
    else:
        logger.log(f'Model saved at {save_dir}',
                   loss=f'{metrics_loss.result():.3f}',
                   best_loss=f'{cp.best_loss:.3f}',
                   )
