import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tensorflow.python.framework.errors_impl import InternalError

import commons

# This script performs
# 1 joining all data-set chunks after translating
# 2 performs model embedding on given data
# 3 splitting into test and train data-sets


def join_dataset(paths):
    df = pd.DataFrame()
    for path in paths:
        df_chunk = pd.read_csv(path)
        df = df.append(df_chunk)
    return df


df_all = join_dataset(('data/original-chunk-0.csv', 'data/original-chunk-1.csv',
                       'data/original-chunk-2.csv', 'data/original-chunk-3.csv',
                       'data/original-chunk-4.csv'))

print('Performing Text Embedding...')

tf.logging.set_verbosity(tf.logging.INFO)
tf.set_random_seed(42)

with tf.Graph().as_default():
    sentences = tf.placeholder(tf.string, name='sentences')
    module = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2", trainable=False)
    embeddings = module(sentences)


    def embed_datasets(sets):
        sess = tf.train.MonitoredSession()
        return (embed_dataset(ds, sess) for ds in sets)


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


    print('preparing datasets')
    embed_output = embed_dataset(df_all, tf.train.MonitoredSession())

df_all['embedding'] = tuple(embed_output['embeddings'].tolist())
df_all = df_all.drop_duplicates('embedding')
print('Gained data-set of', df_all.shape[0], 'elements')


print('Splitting data set to test and train chunks')

df_gas = df_all.loc[df_all['gas'] == True]
df_non_gas = df_all.loc[df_all['gas'] == False]

count_ng = df_non_gas.shape[0]
count_g = df_gas.shape[0]
sum = count_g + count_ng
test = int(sum * 0.2 + 0.5)
train_g = count_g - test
train_sum = train_g + count_ng

print('GAS items:', count_g, 'non GAS items:', count_ng, 'sum:', sum)
print('Test:', test, 'Train:', train_sum, 'Train GAS:', train_g, 'GAS Train %:', train_g / train_sum)

df_test = df_gas[:test]
df_train = df_non_gas.append(df_gas[test:], ignore_index=True)
df_train = commons.shuffle(df_train)

df_test.to_csv('data/test.csv')
df_train.to_csv('data/train.csv')

print('Finished')
