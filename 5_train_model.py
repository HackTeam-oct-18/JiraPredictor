import math

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras import regularizers

import commons

# This script performs
# 1 declaring one or more model
# 2 invoking training for given model(s)
# 3 saving logs for tensorboard

print('Reading data set')
features, labels = commons.read_sequence_training_ds('reduced')

tf.logging.set_verbosity(tf.logging.INFO)
tf.set_random_seed(42)


def try_sequential_model(arch, lr, reg, dropout, name_prefix, activations, loss='mse', batch_size=32, name=None):
    model = keras.Sequential()

    for i in range(len(arch)):
        model.add(layers.Dropout(dropout))
        model.add(layers.Dense(arch[i], activation=activations[i], kernel_regularizer=regularizers.l2(reg)))

    model.build((None, 70))
    model.compile(optimizer=tf.train.AdagradOptimizer(lr),
                  loss=loss,
                  metrics=[metrics.mape, metrics.mae, metrics.mse, commons.mspe])

    if name is None:
        name = '{} units={} activations={} lr={} reg={} dropout={} batch={}'.format(name_prefix, arch, activations,
                                                                                    lr, reg, dropout, batch_size)
    print('Going to train model', name)

    model.fit(features, labels, epochs=250, batch_size=batch_size, validation_split=commons.test_train_ration,
              callbacks=[keras.callbacks.TensorBoard(log_dir='./logs/' + name)])


print('Running training models')
try_sequential_model((36, 1), 5e-2, 3e-2, 0.5, 'linear 1mspe model with p',
                     ('relu', 'linear'), batch_size=32, loss=commons.mspe)
