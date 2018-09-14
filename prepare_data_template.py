import pandas as pd
from numpy import random

gas_sources = ('data/jira-exports/AXNGA.0-4000.980.csv', 'data/jira-exports/AXNGA.4000-8500.965.csv',
               'data/jira-exports/AXNGA.8500-14000.984.csv', 'data/jira-exports/AXNGA.14000-20000.915.csv',
               'data/jira-exports/AXNGA.20000-.101.csv')
non_gas_soures = (
    'data/jira-exports/AXNAU.305.csv', 'data/jira-exports/AXNCEE.823.csv', 'data/jira-exports/AXNIN.9.csv',
    'data/jira-exports/GSGN.66.csv')


def shuffle(df: pd.DataFrame, cols=('key', 'time', 'estimate', 'text')) -> pd.DataFrame:
    vals = df.values
    random.shuffle(vals)
    if cols is not None:
        return pd.DataFrame(columns=cols, data=vals)
    else:
        return pd.DataFrame(vals)


def read_jiras(paths) -> pd.DataFrame:
    df = pd.DataFrame()
    for path in paths:
        df_read = pd.read_csv(path)
        df_chunk = pd.DataFrame()
        df_chunk['key'] = df_read['Issue key']
        df_chunk['time'] = df_read['Σ Time Spent'].values / 3600
        df_chunk['estimate'] = df_read['Σ Original Estimate'].values / 3600
        df_chunk['text'] = (df_read['Summary'] + '\n' + df_read['Description']).values
        df = df.append(df_chunk, ignore_index=True)

    df = df.loc[df['time'] > 0]
    return shuffle(df)


print('Joining DataSets...')

df_gas = read_jiras(gas_sources)
df_non_gas = read_jiras(non_gas_soures)

df_gas.to_csv('data/gas.csv')
df_non_gas.to_csv('data/non_gas.csv')

count_ng = df_non_gas.shape[0]
count_g = df_gas.shape[0]
sum = count_g + count_ng
test = int(sum * 0.2 + 0.5)
train_g = count_g - test
train_sum = train_g + count_ng

print('GAS items:', count_g, 'non GAS items:', count_ng, 'sum:', sum)
print('Test:', test, 'Train:', train_sum, 'Train GAS:', train_g, 'GAS Train %:', train_g / train_sum)

print('Splitting DataSets...')

df_test = df_gas[:test]
df_train = df_non_gas.append(df_gas[test:], ignore_index=True)
df_train = shuffle(df_train)

df_test.to_csv('data/test.csv')
df_train.to_csv('data/train.csv')


print('Finished.')