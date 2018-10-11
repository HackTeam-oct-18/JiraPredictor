from datetime import datetime

import pandas as pd

import commons

translator = commons.create_translator('de', 'en')
print('Adding translations...')


def apply_translate_chain(text, chain):
    for tr in chain:
        tr.set_text(text)
        text = tr.translate()
    return text


def append_translations(df: pd.DataFrame):
    translate_chains = [[commons.create_translator('en', 'de'), commons.create_translator('de', 'en')]]

    all_df = df[0:0]

    for chain in translate_chains:
        print('Applying chain', chain)
        tr_df = df.copy(True)
        length = tr_df.shape[0]
        pct = -2.9
        time = datetime.now()
        for i in range(length):
            new_pct = (100. * i) / length
            if new_pct - pct > 3:
                new_time = datetime.now()
                print('Translated %.2f%%,  ETA %s' % (new_pct,   (new_time - time) * (length - i) / i))
                pct = new_pct
                time = new_time

            src_txt = tr_df['text'][i]
            tr_df['text'][i] = apply_translate_chain(src_txt, chain)

        tr_df = tr_df.loc[tr_df['text'] != df['text']]
        tr_df['original'] = False
        tr_df['lang'] = "en-de/de-en"
        tr_df['service'] = "yandex"
        all_df = all_df.append(tr_df)

    return all_df


df = commons.read_ds('original-chunk-2')
df = append_translations(df)
df.to_csv('data/tr-chunk-2.csv')