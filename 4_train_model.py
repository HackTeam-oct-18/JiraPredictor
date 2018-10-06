import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras import layers
from tensorflow.keras import metrics

import commons

print('Reading data set')
features, labels = commons.read_trainig_ds('data/all.csv')

tf.logging.set_verbosity(tf.logging.INFO)
tf.set_random_seed(42)

print('Building model')

reg = 3e-1
dropout = 0.25

model = keras.Sequential()
model.add(layers.Dropout(dropout))
model.add(layers.Dense(36, activation='relu', kernel_regularizer=regularizers.l2(reg)))
model.add(layers.Dropout(dropout))
model.add(layers.Dense(6, activation='relu', kernel_regularizer=regularizers.l2(reg)))
model.add(layers.Dropout(dropout))
model.add(layers.Dense(1, activation='relu', kernel_regularizer=regularizers.l2(reg)))

# Configure a model for mean-squared error regression.
model.compile(optimizer=tf.train.AdamOptimizer(0.01),
              loss='mse',       # mean squared error
              metrics=[metrics.mae, metrics.mape])

print('Training')

model.fit(features, labels, epochs=200, batch_size=32, validation_split=0.3,
          callbacks=[keras.callbacks.TensorBoard(log_dir='./logs/1')])

print('Finished')
