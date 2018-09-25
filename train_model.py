import itertools

import tensorflow as tf
from tensorflow.python.ops.losses import losses
import numpy as np
import ast
import pandas as pd

test = pd.read_csv('data/test.csv', converters={"embedding": ast.literal_eval})[['embedding', 'time']]
train = pd.read_csv('data/train.csv', converters={"embedding": ast.literal_eval})[['embedding', 'time']]

prepare_ds_fn = lambda df: ({'embeddings': np.array(df['embedding'].tolist())}, df['time'].values)

test = prepare_ds_fn(test)
train = prepare_ds_fn(train)

tf.logging.set_verbosity(tf.logging.INFO)
tf.set_random_seed(42)


learning_rates = (5e-3,)
regularisations = (3e-3,)
dropouts = (.25, )
hiddens = ((6, 5), )

combinations_number = len(learning_rates) * len(regularisations) * len(dropouts) * len(hiddens)
print('Going verify', combinations_number, 'combinations')

min_loss = 100600
best_arch = None
model_number = 0
for arch in hiddens:
    for dropout in dropouts:
        for l2 in regularisations:
            for lr in learning_rates:
                print('Verifying model with arch:', arch, ' reg-l2:', l2, 'dropout: ', dropout, 'lr:', lr)
                columns = [tf.feature_column.numeric_column('embeddings', (512,))]
                estimator = tf.estimator.DNNRegressor(arch, columns, label_dimension=1, dropout=dropout,
                                                      optimizer=tf.train.ProximalAdagradOptimizer(
                                                          learning_rate=lr, l2_regularization_strength=l2),
                                                      loss_reduction=losses.Reduction.MEAN)
                min_model_loss = 100500
                loss_dynamic = []
                for step in range(6):
                    # print('training model')
                    estimator.train(input_fn=lambda: train, steps=500)
                    # print('testing model')
                    results = estimator.evaluate(input_fn=lambda: test, steps=1)
                    # print(results)
                    loss = results['average_loss']
                    loss_dynamic.append(loss)
                    if loss < min_model_loss:
                        min_model_loss = loss
                print('Done verification for model with arch:', arch, ' reg-l2:', l2, 'dropout: ', dropout, 'lr:', lr)
                print('Min model loss:', min_model_loss)
                print(loss_dynamic)
                if min_model_loss < min_loss:
                    min_loss = min_model_loss
                    best_arch = {'arch': arch, 'reg-l2': l2, 'lr': lr, 'dropout': dropout,
                                 'min_loss': min_loss, 'loss_dynamic': loss_dynamic}
                    print("It's the best model from all checked ones", best_arch)
                model_number += 1
                print('Checked {}/{}={:1f}% models'.format(model_number, combinations_number, model_number * 100 / combinations_number))

                if False:
                    print('model evaluation')
                    results = estimator.predict(input_fn=lambda: (train[0], None), yield_single_examples=False)
                    predictions = list(itertools.islice(results, 1))
                    pd.DataFrame(predictions[0]['predictions']).to_csv('data/predictions-train-m{}.csv'.format(model_number))
                    results = estimator.predict(input_fn=lambda: (test[0], None), yield_single_examples=False)
                    predictions = list(itertools.islice(results, 1))
                    pd.DataFrame(predictions[0]['predictions']).to_csv('data/predictions-test-m{}.csv'.format(model_number))
                print('finished')

print('The best model is', best_arch)

