import os
import ast

import numpy as np
import pandas as pd
from numpy import random


def shuffle(df: pd.DataFrame) -> pd.DataFrame:
    vals = df.values
    random.shuffle(vals)
    return pd.DataFrame(columns=df.keys(), data=vals)


def mkdirs(dir):
    os.makedirs(dir, exist_ok=True)


def read_training_ds(path):
    data = pd.read_csv('data/{}.csv'.format(path), converters = {"embedding": ast.literal_eval})[['embedding', 'time']]
    labels = data['time'].values
    data = data['embedding'].values
    
    features = []
    for i in range(data.shape[0]):
        features.append(data[i])
    features = np.asarray(features)
    
    print(features.shape)
    print(labels.shape)
    return features, labels
