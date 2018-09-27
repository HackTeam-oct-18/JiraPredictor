import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from numpy import random
from googletrans import Translator
from tensorflow.python.framework.errors_impl import InternalError

gas_sources = ('data/jira-exports/AXNGA.0-4000.980.csv', 'data/jira-exports/AXNGA.4000-8500.965.csv',
               'data/jira-exports/AXNGA.8500-14000.984.csv', 'data/jira-exports/AXNGA.14000-20000.915.csv',
               'data/jira-exports/AXNGA.20000-.101.csv')
non_gas_soures = (
    'data/jira-exports/AXNAU.305.csv', 'data/jira-exports/AXNCEE.823.csv', 'data/jira-exports/AXNIN.9.csv',
    'data/jira-exports/GSGN.66.csv', 'data/jira-exports/AXON-AXNTH.2.csv', 'data/jira-exports/AXNCN-AXNKR-AXNMY.955.csv',
    'data/jira-exports/AXNASEAN-AXNCH-AXNTW.982.csv')

print('Joining DataSets...')


def shuffle(df: pd.DataFrame) -> pd.DataFrame:
    vals = df.values
    random.shuffle(vals)
    return pd.DataFrame(columns=df.keys(), data=vals)


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
    
    df['text'] = df['text'].apply(lambda text: str(text).strip())
    df = df.loc[df['time'] > 0]
    df = df.loc[df['text'] != '']
    return shuffle(df)


df_gas = read_jiras(gas_sources)
df_non_gas = read_jiras(non_gas_soures)

df_gas.to_csv('data/gas.csv')
df_non_gas.to_csv('data/non_gas.csv')

print('Adding translations...')

def do_translate_chain(text, chain, api):
    for params in chain:
        text = api.translate(text, src=params['src'], dest=params['dest']).text
    return text

def create_with_translation(df: pd.DataFrame):
    translate_chains = (
        ({'src': 'de', 'dest': 'en'},),
        ({'src': 'zh', 'dest': 'en'},),
        ({'src': 'en', 'dest': 'fr'}, {'src': 'fr', 'dest': 'en'},),
        ({'src': 'en', 'dest': 'it'}, {'src': 'it', 'dest': 'en'},),
        ({'src': 'en', 'dest': 'de'}, {'src': 'de', 'dest': 'en'},),
        ({'src': 'en', 'dest': 'es'}, {'src': 'es', 'dest': 'en'},),
        ({'src': 'en', 'dest': 'pt'}, {'src': 'pt', 'dest': 'en'},),
        ({'src': 'en', 'dest': 'nl'}, {'src': 'nl', 'dest': 'en'},),
        ({'src': 'en', 'dest': 'da'}, {'src': 'da', 'dest': 'en'},),
        ({'src': 'en', 'dest': 'no'}, {'src': 'no', 'dest': 'en'},),
        ({'src': 'en', 'dest': 'sv'}, {'src': 'sv', 'dest': 'en'},),
        ({'src': 'en', 'dest': 'cs'}, {'src': 'cs', 'dest': 'en'},),
        ({'src': 'en', 'dest': 'pl'}, {'src': 'pl', 'dest': 'en'},),
        ({'src': 'en', 'dest': 'ru'}, {'src': 'ru', 'dest': 'en'},),
        ({'src': 'en', 'dest': 'ro'}, {'src': 'ro', 'dest': 'en'},),
        ({'src': 'en', 'dest': 'el'}, {'src': 'el', 'dest': 'en'},),
        ({'src': 'en', 'dest': 'zh'}, {'src': 'zh', 'dest': 'en'},),
        ({'src': 'en', 'dest': 'sk'}, {'src': 'sk', 'dest': 'en'},),
        ({'src': 'en', 'dest': 'fy'}, {'src': 'fy', 'dest': 'en'},),
        ({'src': 'en', 'dest': 'ja'}, {'src': 'ja', 'dest': 'en'},),
        ({'src': 'de', 'dest': 'en'}, {'src': 'en', 'dest': 'fr'}, {'src': 'fr', 'dest': 'en'},),
        ({'src': 'zh', 'dest': 'en'}, {'src': 'en', 'dest': 'fr'}, {'src': 'fr', 'dest': 'en'},),
        ({'src': 'de', 'dest': 'en'}, {'src': 'en', 'dest': 'nl'}, {'src': 'nl', 'dest': 'en'},),
        ({'src': 'zh', 'dest': 'en'}, {'src': 'en', 'dest': 'nl'}, {'src': 'nl', 'dest': 'en'},),
        ({'src': 'de', 'dest': 'en'}, {'src': 'en', 'dest': 'sv'}, {'src': 'sv', 'dest': 'en'},),
        ({'src': 'de', 'dest': 'en'}, {'src': 'en', 'dest': 'it'}, {'src': 'it', 'dest': 'en'},),
        ({'src': 'de', 'dest': 'en'}, {'src': 'en', 'dest': 'da'}, {'src': 'da', 'dest': 'en'},),
        ({'src': 'zh', 'dest': 'en'}, {'src': 'en', 'dest': 'da'}, {'src': 'da', 'dest': 'en'},),
        ({'src': 'de', 'dest': 'en'}, {'src': 'en', 'dest': 'el'}, {'src': 'da', 'dest': 'en'},),
        ({'src': 'de', 'dest': 'en'}, {'src': 'en', 'dest': 'no'}, {'src': 'no', 'dest': 'en'},),
    )
	
    df['original'] = True
    df['lang'] = 'en'
    translated_rows = []
    text_pos = df.keys().get_loc('text')
    original_pos = df.keys().get_loc('original')
    lang_pos = df.keys().get_loc('lang')
    api = Translator(timeout=30, service_urls=[
      'translate.google.com',
      'translate.google.co.kr',
    ])
    for row in df.values:
        row[original_pos] = False
        src_text = row[text_pos]
        translated_texts = {src_text}
        for chain in translate_chains:
            print('creating translations for chain', chain)
            new_text = do_translate_chain(src_text, api=api, chain=chain)
            if not translated_texts.__contains__(new_text):
                translated_texts.add(new_text)
                new_row = row[:]
                new_row[text_pos] = new_text[:]
                new_row[lang_pos] = chain[:]
                translated_rows.append(new_row)
    return df.append(pd.DataFrame(translated_rows, columns=df.keys()))

print('adding translations for non gas')
df_non_gas = create_with_translation(df_non_gas)
df_non_gas = shuffle(df_non_gas)
print('adding translations for gas')
df_gas = create_with_translation(df_gas)
df_gas = shuffle(df_gas)

df_gas.to_csv('data/gas-with-translations.csv')
df_non_gas.to_csv('data/non-gas-with-translations.csv')

print('Splitting DataSets...')

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
df_train = shuffle(df_train)

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
                print('wtf detected')
                df = pd.DataFrame(chunk)
                df.to_csv('wtf.csv')
                quit()
            embeds = np.append(embeds, chunk, 0)
            print(embeds.shape)
        return {'embeddings': embeds}


    print('preparing datasets')
    train, test = embed_datasets((df_train, df_test))

print('Saving Model...')

df_test['embedding'] = tuple(test['embeddings'].tolist())
df_train['embedding'] = tuple(train['embeddings'].tolist())

df_test.to_csv('data/test.csv')
df_train.to_csv('data/train.csv')

print('Finished.')
