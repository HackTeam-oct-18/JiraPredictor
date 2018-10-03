import os


import pandas as pd
from yandex_translate import YandexTranslate

from commons import shuffle

print('Adding translations...')
print('Using YandexTranslate API key', os.environ['Y_KEY'])
translate = YandexTranslate(os.environ['Y_KEY'])


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

