import re

import pandas as pd

import commons

# This script performs
# 1 joining all data from Jira to single dataset
# 2 filter and pre-process input data
# 3 save data set
# 4 splitting data-set to chunks for translation


TRANSLATOR_TEXT_LIMIT = 1_000_000
TEXT_LENGTH_MAX_LIMIT = 512
TEXT_LENGTH_MIN_LIMIT = 8

gas_sources = ('data/jira-exports/AXNGA.0-4000.980.csv', 'data/jira-exports/AXNGA.4000-8500.965.csv',
               'data/jira-exports/AXNGA.8500-14000.984.csv', 'data/jira-exports/AXNGA.14000-20000.915.csv',
               'data/jira-exports/AXNGA.20000-.101.csv')
non_gas_soures = (
    'data/jira-exports/AXNAU.305.csv', 'data/jira-exports/AXNCEE.823.csv', 'data/jira-exports/AXNIN.9.csv',
    'data/jira-exports/GSGN.66.csv', 'data/jira-exports/AXON-AXNTH.2.csv',
    'data/jira-exports/AXNCN-AXNKR-AXNMY.955.csv',
    'data/jira-exports/AXNASEAN-AXNCH-AXNTW.982.csv')

print('Joining DataSets...')

table_pattern = re.compile("^\|\|.+\|\|($[\r\n(\r\n)]{1}^\|.+\|$)+[\r\n(\r\n)]?", re.MULTILINE)
eol_pattern = re.compile("[\n\r(\r\n)]{2,}", re.MULTILINE)
spaces_pattern = re.compile("([ ]{2,})|(\t+)|([ \t]{2,})")


def preprocess_text(text):
    text = str(text)
    text = table_pattern.sub("", text)
    text = eol_pattern.sub("\n", text)
    text = spaces_pattern.sub(" ", text)
    text = text.strip()

    # TODO: Remove links
    # TODO: Build histogramm of text length after all
    if len(text) > TEXT_LENGTH_MAX_LIMIT:
        # suppose that important information is at the text start and at the text end
        tail = int(TEXT_LENGTH_MAX_LIMIT * .15 + .5)
        head = TEXT_LENGTH_MAX_LIMIT - tail - 1
        return text[:head] + ' ' + text[-tail:]
    if len(text) < TEXT_LENGTH_MIN_LIMIT:
        return ''
    return text


priorities_names = {'SOS': 'P0', 'Critical': 'P1', 'High': 'P2',
                    'Medium': 'P3', 'Low': 'P4', 'Undefined': 'PU'}


def read_jiras(paths) -> pd.DataFrame:
    df = pd.DataFrame()
    for path in paths:
        df_read = pd.read_csv(path)
        df_chunk = pd.DataFrame()
        # TODO: Check what data would be useful for model (priority, labels)
        df_chunk['key'] = df_read['Issue key']
        df_chunk['project_id'] = df_read['Project key']
        df_chunk['priority'] = df_read['Priority'].apply(lambda p: priorities_names[p])
        df_chunk['time'] = df_read['Σ Time Spent'].values / 3600
        df_chunk['estimate'] = df_read['Σ Original Estimate'].values / 3600
        df_chunk['text'] = (df_read['Summary'] + '\n' + df_read['Description']).values
        df = df.append(df_chunk, ignore_index=True)

    df['text'] = df['text'].apply(preprocess_text)
    df = df.loc[df['time'] > 0]
    df = df.loc[df['time'] <= 40]
    df = df.loc[df['text'] != '']
    df['original'] = True
    df['lang'] = 'en'
    return commons.shuffle(df)


df_gas = read_jiras(gas_sources)
df_non_gas = read_jiras(non_gas_soures)

df_all = df_gas.append(df_non_gas, ignore_index=True)
df_all = commons.shuffle(df_all)
print('Gained data set of', df_all.shape[0], 'elements')

#######

print('Saving all data set')
df_all.to_csv('data/text.csv')

#######

print('Splitting data-set for translation with limit up to', TRANSLATOR_TEXT_LIMIT, 'symbols...')

text_pos = df_all.keys().get_loc('text')

chunk_number = 0
chunk_text_length = 0
chunk = pd.DataFrame(columns=df_all.keys())
for row in df_all.values[:]:
    new_text_length = len(row[text_pos])
    new_chunk_text_length = chunk_text_length + new_text_length
    row = row.reshape(1, len(df_all.keys()))
    if new_chunk_text_length <= TRANSLATOR_TEXT_LIMIT:
        chunk_text_length = new_chunk_text_length
        chunk = chunk.append(pd.DataFrame(row, columns=df_all.keys()))
    else:
        print('saving', chunk_number, 'chunk with overall text length', chunk_text_length, 'and overall row number',
              chunk.shape[0])
        chunk.to_csv('data/original-chunk-{}.csv'.format(chunk_number))
        chunk_number += 1
        chunk_text_length = new_text_length
        chunk = pd.DataFrame(columns=df_all.keys(), data=row)

print('saving final', chunk_number, 'chunk with overall text length', chunk_text_length, 'and overall row number',
      chunk.shape[0])
chunk.to_csv('data/original-chunk-{}.csv'.format(chunk_number))
