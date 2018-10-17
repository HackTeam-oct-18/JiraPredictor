import os
import ast
import glob

from tensorflow.python.keras import backend as K
from tensorflow.python.ops import math_ops
import numpy as np
import pandas as pd
from numpy import random
import yandex

# common project utilities

test_train_ration = 0.2


def shuffle(df: pd.DataFrame) -> pd.DataFrame:
    vals = df.values
    random.shuffle(vals)
    return pd.DataFrame(columns=df.keys(), data=vals)


def mkdirs(dir):
    os.makedirs(dir, exist_ok=True)


def read_ds(name):
    ds = pd.read_csv('data/{}.csv'.format(name), converters={"embedding": ast.literal_eval,
                                                             "reduced_embedding": ast.literal_eval})
    return ds.sort_values('original') # translated ones will be in train set


def read_sequence_training_ds(name, inputs=('reduced_embedding', 'priority_val'), is_shuffle=False):
    data = read_ds(name)[['reduced_embedding', 'time', 'original', 'priority_val', 'priority_unknown']]
    if is_shuffle:
        data = shuffle(data)
    data = data.sort_values('original')  # translated ones will be in train set
    labels = data['time'].values
    embeds = expand_nparray_of_lists(data['reduced_embedding'].values)
    priors = data[['priority_val', 'priority_unknown']].values # looks like doesn't work
    features = np.append(embeds, priors, 1)
    # features = embeds

    print(features.shape)
    print(labels.shape)
    return features, labels


def expand_nparray_of_lists(nparr):
    rows = []
    for i in range(nparr.shape[0]):
        rows.append(nparr[i])
    return np.asarray(rows)


def join_dataset(name_regex, is_shuffle=False):
    df = pd.DataFrame()
    for name in glob.glob('data/%s.csv' % name_regex):
        df_chunk = pd.read_csv(name)
        df = df.append(df_chunk)

    if is_shuffle:
        return shuffle(df)
    df = pd.DataFrame(data=df.values, columns=df.keys())
    return df


def create_translator(src, dest) -> yandex.Translater:
    translator = yandex.Translater()
    translator.set_key(os.environ['Y_KEY'])
    translator.set_from_lang(src)
    translator.set_to_lang(dest)

    return translator


def compose(upper_fn, inner_fn):
    return lambda x: upper_fn(inner_fn(x))


def create_mspemse(mse_div, scale=100.):
    def mspemse(y_true, y_pred):
        mse = math_ops.square(y_true - y_pred)
        mspe = mse / K.clip(math_ops.square(y_true), K.epsilon(), None)
        return scale * K.mean(mspe + mse / mse_div, axis=-1)

    return mspemse


def create_mapemae(mae_div, scale=100.):
    def mapemae(y_true, y_pred):
        mae = math_ops.abs(y_true - y_pred)
        mape = mae / K.clip(math_ops.abs(y_true), K.epsilon(), None)
        return scale * K.mean(mape + mae / mae_div, axis=-1)

    return mapemae


def mean_squared_percentage_error(y_true, y_pred):
    diff = math_ops.square(
        (y_true - y_pred) / K.clip(y_true, K.epsilon(), None))
    return 100. * K.mean(diff, axis=-1)


mspe = mean_squared_percentage_error
