import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd

test = pd.read_csv('data/test.csv')[['text', 'time']]
train = pd.read_csv('data/train.csv')[['text', 'time']]

tf.logging.set_verbosity(tf.logging.INFO)

with tf.Graph().as_default():
    sentences = tf.placeholder(tf.string)
    time = tf.placeholder(tf.float32)
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

print('datasets prepared, creating regression model')
columns = [tf.feature_column.numeric_column('embeddings', (512,))]
estimator = tf.estimator.DNNRegressor([6, 6], columns, label_dimension=1, dropout=.5,
                                      optimizer=tf.train.ProximalAdagradOptimizer(
                                          learning_rate=0.1, l2_regularization_strength=0.03))
min_loss = 2.
loss = 100500.
while loss > min_loss:
    print('training model')
    estimator.train(input_fn=lambda: train, steps=1000)
    print('testing model')
    results = estimator.evaluate(input_fn=lambda: test, steps=1)
    print(results)
    loss = results['average_loss']
print('model showcase')
results = estimator.predict(input_fn=lambda: (train[0], None), yield_single_examples=False)
print(results)
print(list(results))
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
