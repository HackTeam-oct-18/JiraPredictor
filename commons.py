import os
import ast

import numpy as np
import pandas as pd
from numpy import random
import yandex


def shuffle(df: pd.DataFrame) -> pd.DataFrame:
    vals = df.values
    random.shuffle(vals)
    return pd.DataFrame(columns=df.keys(), data=vals)


def mkdirs(dir):
    os.makedirs(dir, exist_ok=True)


def read_ds(name):
    ds = pd.read_csv('data/{}.csv'.format(name), converters={"embedding": ast.literal_eval})
    return ds


def read_training_ds(name, is_shuffle=False):
    data = read_ds(name)[['embedding', 'time']]
    if is_shuffle:
        data = shuffle(data)
    labels = data['time'].values
    features = expand_nparray_of_lists(data['embedding'].values)
    print(features.shape)
    print(labels.shape)
    return features, labels


def expand_nparray_of_lists(nparr):
    rows = []
    for i in range(nparr.shape[0]):
        rows.append(nparr[i])
    return np.asarray(rows)


def join_dataset(paths, is_shuffle=False):
    df = pd.DataFrame()
    for path in paths:
        df_chunk = pd.read_csv(path)
        df = df.append(df_chunk)

    if is_shuffle:
        return shuffle(df)
    return df


def create_translator(src, dest) -> yandex.Translater:
    translator = yandex.Translater()
    translator.set_key(os.environ['Y_KEY'])
    translator.set_from_lang(src)
    translator.set_to_lang(dest)

    return translator
