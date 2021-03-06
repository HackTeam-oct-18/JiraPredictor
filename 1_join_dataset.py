import re

import pandas as pd

import commons

# This script performs
# 1 joining all data from Jira to single dataset
# 2 filter and pre-process input data
# 3 for long texts in Jira drops text's center in with different head-tail chars ratio
# 4 save data set
# 5 splitting data-set to chunks for translation


TRANSLATOR_TEXT_LIMIT = 1_000_000 / 2
TEXT_LENGTH_MAX_LIMIT = 512
TEXT_LENGTH_MIN_LIMIT = 8

sources = [('data/jira-exports/%d.csv') % i for i in range(1, 40)]

print('Joining DataSets...')

table_pattern = re.compile("^\|\|.+\|\|($[\r\n(\r\n)]{1}^\|.+\|$)+[\r\n(\r\n)]?", re.MULTILINE)
eol_pattern = re.compile("[\n\r(\r\n)]{2,}", re.MULTILINE)
spaces_pattern = re.compile("([ ]{2,})|(\t+)|([ \t]{2,})")
code_n_color_pattern = re.compile("{((code)|(color))(:[a-zA-Z0-9#]+)?}")

priorities_names = {'SOS': 'P0', 'Critical': 'P1', 'High': 'P2',
                    'Medium': 'P3', 'Minor': 'P3', 'Low': 'P4', 'Disagree': 'P4', 'Undefined': 'PU'}


def preprocess_text(text):
    text = str(text)
    text = table_pattern.sub("", text)
    text = code_n_color_pattern.sub("", text)
    text = eol_pattern.sub("\n", text)
    text = spaces_pattern.sub(" ", text)
    text = text.strip()

    # TODO: Remove links
    # TODO: Build histogramm of text length after all
    if len(text) < TEXT_LENGTH_MIN_LIMIT:
        return ''
    return text


def create_cut_center(head_tail_ratio):
    def cut_center(text):
        if len(text) > TEXT_LENGTH_MAX_LIMIT:
            head = int(TEXT_LENGTH_MAX_LIMIT * head_tail_ratio + .5)
            tail = TEXT_LENGTH_MAX_LIMIT - head - 1

            if tail > 0 and head > 0:
                return text[:head] + '.' + text[-tail:]
            else:
                if tail <= 0:
                    return text[:TEXT_LENGTH_MAX_LIMIT]
                else:
                    return text[-TEXT_LENGTH_MAX_LIMIT:]
        return text
    return cut_center


def read_jiras(paths) -> pd.DataFrame:
    df = pd.DataFrame()
    for path in paths:
        df_read = pd.read_csv(path)
        df_chunk = pd.DataFrame()
        # TODO: Check what data would be useful for model (priority, labels)
        df_chunk['key'] = df_read['Issue key']
        if 'Component/s' not in df_read:
            df_read['Component/s'] = ''
        df_chunk['component'] = df_read['Component/s']
        if 'Project key' not in df_read:
            df_read['Project key'] = ''
        df_chunk['project_id'] = df_read['Project key']
        df_chunk['reporter'] = df_read['Reporter']
        if 'Labels' not in df_read:
            df_read['Labels'] = ''
        if 'Description' not in df_read:
            df_read['Description'] = ''
        if 'Σ Time Spent' not in df_read:
            df_read['Σ Time Spent'] = 0
        if 'Σ Original Estimate' not in df_read:
            df_read['Σ Original Estimate'] = 0
        df_chunk['label'] = df_read['Labels']
        df_chunk['priority'] = df_read['Priority'].apply(lambda p: priorities_names[p])
        df_chunk['time'] = df_read['Σ Time Spent'].values / 3600
        df_chunk['estimate'] = df_read['Σ Original Estimate'].values / 3600
        df_chunk['text'] = (df_read['Summary'] + '\n' + df_read['Description']).values
        # Project key

        df = df.append(df_chunk, ignore_index=True)

    df['text'] = df['text'].apply(preprocess_text)
    df = df.loc[df['time'] > 0]
    df = df.loc[df['time'] <= 40]
    df = df.loc[df['text'] != '']
    df['original'] = True
    df['lang'] = 'en'

    # take different parts of long text, will increase data
    mult_df = df[0:0]
    for ratio in (.15, 0.5, .85):
        df_tmp = df[:]
        df_tmp['text'] = df_tmp['text'].apply(create_cut_center(ratio))
        mult_df = mult_df.append(df_tmp)
        df['keep_head_tail_ration'] = ratio

    return commons.shuffle(mult_df)


df_all = read_jiras(sources)

size = df_all.shape[0]
print('Dropping duplicates')
df_all = df_all.drop_duplicates('text')
print('Gained data set of {} jiras, {} ones were filtered as duplicates'.format(df_all.shape[0],
                                                                                             size - df_all.shape[0]))
df_all.to_csv('data/trace/all_original.csv')

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
        print('saving', chunk_number, 'chunk with overall text length', chunk_text_length, 'and overall rows number',
              chunk.shape[0])
        chunk.to_csv('data/original-chunk-{}.csv'.format(chunk_number))
        chunk_number += 1
        chunk_text_length = new_text_length
        chunk = pd.DataFrame(columns=df_all.keys(), data=row)

print('saving', chunk_number, 'chunk with overall text length', chunk_text_length, 'and overall rows number',
      chunk.shape[0])
chunk.to_csv('data/original-chunk-{}.csv'.format(chunk_number))
