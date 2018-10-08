import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras import layers
from tensorflow.keras import metrics

import commons

print('Reading data set')
features, labels = commons.read_training_ds('all')

tf.logging.set_verbosity(tf.logging.INFO)
tf.set_random_seed(42)


def try_model(arch, lr, reg, dropout, name_prefix, activation='relu', name=None):
    model = keras.Sequential()

    for units in arch:
        model.add(layers.Dropout(dropout))
        model.add(layers.Dense(units, activation=activation, kernel_regularizer=regularizers.l2(reg)))

    model.build((None, 512))
    model.compile(optimizer=tf.train.AdagradOptimizer(lr),
                  loss='mse',       # mean squared error
                  metrics=[metrics.mape, metrics.mae])

    if name is None:
        name = '{} arch={} lr={} reg={} dropout={} {}'.format(name_prefix, arch, lr, reg, dropout, activation)
    print('Going to train model', name)

    model.fit(features, labels, epochs=250, batch_size=32, validation_split=0.2,
              callbacks=[keras.callbacks.TensorBoard(log_dir='./logs/' + name)])


print('Running training models')
print("You can run command 'tensorboard --logdir ./logs' to view taraining process in browser")
try_model((6, 6, 1), 1e-2, 3e-3, 0.35, 'basic model')
try_model((6, 6, 1), 1e-2, 3e-3, 0.35, 'basic model', 'elu')
try_model((6, 6, 1), 1e-2, 3e-3, 0.35, 'basic model', 'sigmoid')
