import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras import metrics
from tensorflow.keras import regularizers

import commons

# This script performs
# 1 declaring one or more model
# 2 invoking training for given model(s)
# 3 saving logs for tensorboard

print('Reading data set')
features, labels = commons.read_training_ds('reduced')

tf_board_log_dir_root = './logs/'
model_checkpoints_dir_root = './models_cache/chkpts/'
commons.mkdirs(tf_board_log_dir_root)
commons.mkdirs(model_checkpoints_dir_root)

tf.logging.set_verbosity(tf.logging.INFO)
tf.set_random_seed(42)


def try_model(arch, lr, reg, dropout, name_prefix, activations, loss='mse', batch_size=32, name=None):
    model = keras.Sequential()

    for i in range(len(arch)):
        model.add(layers.Dropout(dropout))
        model.add(layers.Dense(arch[i], activation=activations[i], kernel_regularizer=regularizers.l2(reg)))

    model.build((None, 64))
    model.compile(optimizer=tf.train.AdagradOptimizer(lr),
                  loss=loss,
                  metrics=[metrics.mape, metrics.mae, metrics.mean_squared_error])

    if name is None:
        name = '{} units={} #{}'.format(name_prefix, arch, int(time.time() + 0.5))
    print('Going to train model', name)

    model.fit(features, labels, epochs=250, batch_size=batch_size, validation_split=commons.test_train_ration,
              callbacks=[keras.callbacks.TensorBoard(log_dir= tf_board_log_dir_root + name),
                         keras.callbacks.ModelCheckpoint(model_checkpoints_dir_root + name + '-best.cpt', save_best_only=True),
                         keras.callbacks.ModelCheckpoint(model_checkpoints_dir_root + name + '-last.cpt')])


print('Running training models')
try_model((64, 18, 1), 1e-3, 1e-3, 0.15, 'exp deep mape-mae model',
          ('elu', 'elu', tf.exp), batch_size=32,
          loss=commons.create_mapemae(np.max(labels)))
