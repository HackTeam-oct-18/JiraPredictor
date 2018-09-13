import pandas as pd
from numpy import random

gas_sources = ('data/jira-exports/AXNGA.0-4000.980.csv', 'data/jira-exports/AXNGA.4000-8500.965.csv', 'data/jira-exports/AXNGA.8500-14000.984.csv', 'data/jira-exports/AXNGA.14000-20000.915.csv', 'data/jira-exports/AXNGA.20000-.101.csv')


def read_jiras(paths) -> pd.DataFrame:
    df = pd.DataFrame()
    for path in paths:
        df_read = pd.read_csv(path)
        df_chunk = pd.DataFrame()
        df_chunk['text'] = (df_read['Summary'] + '\n' + df_read['Description']).values
        df_chunk['time'] = df_read['Σ Time Spent'].values
        df = df.append(df_chunk, ignore_index=True)

    vals = df.values
    random.shuffle(vals)
    return pd.DataFrame(columns=['text', 'time'], data=vals)


df_gas = read_jiras(gas_sources)
print(df_gas)

df_gas.to_csv('data/gas.csv')