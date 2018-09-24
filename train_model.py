import itertools

import tensorflow as tf
from tensorflow.python.ops.losses import losses
import tensorflow_hub as hub
import numpy as np
import pandas as pd

test = pd.read_csv('data/test.csv')[['text', 'time']]
train = pd.read_csv('data/train.csv')[['text', 'time']]

tf.logging.set_verbosity(tf.logging.INFO)
tf.set_random_seed(42)

with tf.Graph().as_default():
    sentences = tf.placeholder(tf.string, name='sentences')
    time = tf.placeholder(tf.float32, name='time')
    module = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2", trainable=False)
    embeddings = module(sentences)

    def embed_datasets(sets):
        print('initing session')
        sess = tf.train.MonitoredSession()
        return (embed_dataset(ds, sess) for ds in sets)

    def embed_dataset(ds: pd.DataFrame, sess):
        step = 128
        embeds = np.zeros((0, 512))
        print('invoking embedding with chunks <=', step, 'for', ds.shape[0], 'sentences')
        for start in range(0, ds.shape[0], step):
            end = start + step
            chunk = ds[start:end]['text'].values
            chunk = sess.run(embeddings, {sentences: chunk})
            embeds = np.append(embeds, chunk, 0)
            print(embeds.shape)
        return {'embeddings': embeds}, ds['time'].values

    print('preparing datasets')
    train, test = embed_datasets((train, test))


learning_rates = (5e-3, 1e-2, 5e-2)
regularisations = (3e-1, 1., 3., 1e+1)
dropouts = (.15, )
hiddens = ((64, 18), )

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

print('The best model is', best_arch)

if False:
    print('model showcase')
    # train[0]['embeddings'] = train[0]['embeddings'][:5]
    results = estimator.predict(input_fn=lambda: (train[0], None), yield_single_examples=False)
    # print(train[0]['embeddings'].shape[1])
    predictions = list(itertools.islice(results, 1))
    print(predictions)
print('finished')

# saver = tf.train.Saver()
#
# print(sess.run(estimate, {sentences: texts}))

# def make_embed_fn(module_name):
#     with tf.Graph().as_default():
#         sentences = tf.placeholder(tf.string)
#         embed = hub.Module(module_name)
#         embeddings = embed(sentences)
#         session = tf.train.MonitoredSession()
#     return lambda x: session.run(embeddings, {sentences: x})

#
# f2 = make_embed_fn("https://tfhub.dev/google/universal-sentence-encoder/2")
# f2_dup = make_embed_fn("https://tfhub.dev/google/universal-sentence-encoder/2")
#
# x = [
#     "The quick brown fox jumps over the lazy dog.",
#     "I am a sentence for which I would like to get its embedding",
# ]
#
# print('Done two functions')
# print(f2)
# print(f2_dup)
#
# # Returns zeros showing the module is stable across instantiations.
# print(np.linalg.norm(f2(x) - f2_dup(x)))
#
# # ISSUE: returns 0.3 showing the embeddings per example depend on other elements in the batch.
# print(np.linalg.norm(f2(x[0:1]) - f2(x)[0:1]))
#
# print("let's do it")
#
