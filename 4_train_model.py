import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.python.ops import math_ops
from tensorflow.keras import metrics
from tensorflow.python.keras import backend as K

import commons

print('Reading data set')
features, labels = commons.read_training_ds('all')

tf.logging.set_verbosity(tf.logging.INFO)
tf.set_random_seed(42)


def exp(x):
    return tf.exp(x)


def mapemae(y_true, y_pred):
    rel_diff = math_ops.abs(
        (y_true - y_pred) / K.clip(math_ops.abs(y_true), K.epsilon(), None))
    abs_diff = math_ops.abs(
        (y_true - y_pred) / 6.1)
    diff = (rel_diff + abs_diff) * .5
    return 100. * K.mean(diff, axis=-1)


def try_model(arch, lr, reg, dropout, name_prefix, activations, loss='mse', batch_size=32, name=None):
    model = keras.Sequential()

    for i in range(len(arch)):
        model.add(layers.Dropout(dropout))
        model.add(layers.Dense(arch[i], activation=activations[i], kernel_regularizer=regularizers.l2(reg)))

    model.build((None, 512))
    model.compile(optimizer=tf.train.AdagradOptimizer(lr),
                  loss=loss,
                  metrics=[metrics.mape, metrics.mae])

    if name is None:
        name = '{} units={} activations={} lr={} reg={} dropout={} batch={}'.format(name_prefix, arch, activations,
                                                                                    lr, reg, dropout, batch_size)
    print('Going to train model', name)

    model.fit(features, labels, epochs=500, batch_size=batch_size, validation_split=0.2,
              callbacks=[keras.callbacks.TensorBoard(log_dir='./logs/' + name)])


print('Running training models')
try_model((6, 6, 1), 1e-3, 1e-3, 0.35, 'exp deep  model mape-mae',
          ('tanh', 'tanh', exp, 'linear'), batch_size=16, loss=mapemae)
