from os import environ
import sys

import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tensorflow.python.framework.errors_impl import InternalError

import commons

# This script performs
# 1 joining all data-set given chunks
# 2 performs model embedding on given data
# 3 binarize priority


if environ.get('TFHUB_CACHE_DIR') is None:
    print("WARNING: you haven't provide TFHUB_CACHE_DIR system variable, model will be downloaded to temp folder.")

print('Reading data set')
df_all = commons.join_dataset('original-chunk-*')

tf.logging.set_verbosity(tf.logging.INFO)
tf.set_random_seed(42)


def add_priorities(df: pd.DataFrame):
    priority_val_map = {'P0': 1., 'P1': .33, 'P2': 0., 'PU': 0., 'P3': -.33, 'P4': -1.}
    for row in range(df.shape[0]):
        priority = df['priority'][row]
        df.loc[row, 'priority_val'] = priority_val_map[priority]
        if priority == 'PU':
            df.loc[row, 'priority_unknown'] = 1.
        else:
            df.loc[row, 'priority_unknown'] = 0.

    return df


def hash_string(text: str):
    hash = 0
    if text != '':
        hash = text.__hash__()
    return hash / sys.maxsize


def add_hash(df, col):
    new_col = col + '_hash'
    df[new_col] = df[col].apply(hash_string)
    df[new_col] = (df[new_col] - df[new_col].mean()) / df[new_col].max()


def add_is_region(df, regionnnn):
    def mapper(reg):
        if reg == regionnnn:
            return 1.
        else:
            return 0.

    df['is_' + regionnnn] = df['project_id'].apply(mapper)


print('Adding priority labels')
df_all = add_priorities(df_all)

print('Adding hashes')
for col in ('component', 'project_id', 'reporter', 'label'):
    add_hash(df_all, col)

print('Adding is region')
for region in ['AXNASEAN', 'AXNCH', 'AXNTW', 'AXNAU', 'AXNCEE', 'AXNCN', 'AXNKR', 'AXNMY',
               'AXNGA', 'AXNIN', 'AXON', 'AXNTH', 'GSGN']:
    add_is_region(df_all, region)

######

print('Adding text embedding')
with tf.device('/cpu:0'):
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

    print('Performing Text Embedding...')
    embed_output = embed_dataset(df_all, tf.train.MonitoredSession())

df_all['embedding'] = tuple(embed_output['embeddings'].tolist())
size = df_all.shape[0]
# print('Dropping duplicates')
# df_all = df_all.drop_duplicates('embedding')
print('Gained data set of {} embedding elements, {} ones were filtered as duplicates'.format(df_all.shape[0],
                                                                                             size - df_all.shape[0]))
#######

print('Saving data set with embedding')
df_all.to_csv('data/embedded.csv')
