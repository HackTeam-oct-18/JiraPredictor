from os import environ

import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tensorflow.python.framework.errors_impl import InternalError

import commons

# This script performs
# 1 joining all data-set chunks after translating
# 2 performs model embedding on given data


if environ.get('TFHUB_CACHE_DIR') is None:
    print("WARNING: you haven't provide TFHUB_CACHE_DIR system variable, model will be downloaded to temp folder.")

df_all = commons.join_dataset(('data/original-chunk-0.csv', 'data/original-chunk-1.csv',
                               'data/original-chunk-2.csv'))

print('Performing Text Embedding...')

tf.logging.set_verbosity(tf.logging.INFO)
tf.set_random_seed(42)

with tf.Graph().as_default():
    sentences = tf.placeholder(tf.string, name='sentences')
    module = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2", trainable=False)
    embeddings = module(sentences)


    def embed_dataset(ds: pd.DataFrame, sess):
        step = 128
        embeds = np.zeros((0, 512))
        print('invoking embedding with chunks <=', step, 'for', ds.shape[0], 'sentences')
        for start in range(0, ds.shape[0], step):
            end = start + step
            chunk = ds[start:end]['text'].values
            try:
                chunk = sess.run(embeddings, {sentences: chunk})
            except InternalError:
                print('wtf has been detected and saved to wtf.csv')
                ds[start:end].to_csv('wtf.csv')
                quit()
            embeds = np.append(embeds, chunk, 0)
            print(embeds.shape)
        return {'embeddings': embeds}


    print('preparing data-sets')
    embed_output = embed_dataset(df_all, tf.train.MonitoredSession())

df_all['embedding'] = tuple(embed_output['embeddings'].tolist())
size = df_all.shape[0]
df_all = df_all.drop_duplicates('embedding')
print('Gained data set of {} embedding elements, {} ones were filtered as duplicates'.format(df_all.shape[0],
                                                                                             size - df_all.shape[0]))

#######

print('Saving data set with embedding')
df_all.to_csv('data/all.csv')
