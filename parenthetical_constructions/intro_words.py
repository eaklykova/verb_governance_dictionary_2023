"""
Код для выделения вводных конструкций из синтаксической разметки deeppavlov.
"""

import string
import pickle
from collections import Counter
import pandas as pd
from conllu import parse
from nltk.corpus import stopwords
from minio import Minio
from tqdm.auto import tqdm


def get_minio_filenames(client):
    """
    Получает список файлов в папке syntax-parsed и их размеры.
    :param client: minio client
    :return: список кортежей вида (имя файла, размер файла)
    """
    # получаем список файлов и сортируем по возрастанию размера
    syntax_parsed = client.list_objects(bucket_name='public',
                                        prefix='syntax-parsed/')
    files1 = [(obj.object_name, obj.size) for obj in syntax_parsed]
    bondarev = client.list_objects(bucket_name='public',
                                   prefix='syntax-parsed/Bondarev/')
    files2 = [(obj.object_name, obj.size) for obj in bondarev]
    conllus = files1 + files2
    files = [(file[0], round(file[-1] / 2**30, 2))
             for file in conllus if file[0].endswith('.conllu')]
    files.sort(key=lambda x: x[-1])
    return files


def get_comma_ngrams(sentences):
    """
    Собирает из conllu-разметки n-граммы, находящиеся между запятыми
    или в начале предложения перед запятой.
    :param sentences: предложения в виде conllu-разметки
    :return: список n-грамм из всех предложений
    """
    ngrams = []
    for sentence in tqdm(sentences):
        # идем от начала предложения
        try:
            if sentence[1]['form'] == ',':
                ngrams.append(sentence[0:1])
            elif sentence[2]['form'] == ',':
                ngrams.append(sentence[0:2])
            elif sentence[3]['form'] == ',':
                ngrams.append(sentence[0:3])
        except IndexError:
            pass
        for i, token in enumerate(sentence):
            # первый токен уже обработан
            if i == 0:
                continue
            # идем от запятой
            if token['form'] == ',':
                try:
                    if sentence[i+2]['form'] == ',':
                        ngrams.append(sentence[i+1:i+2])
                    elif sentence[i+3]['form'] == ',':
                        ngrams.append(sentence[i+1:i+3])
                    elif sentence[i+4]['form'] == ',':
                        ngrams.append(sentence[i+1:i+4])
                except IndexError:
                    pass
    return ngrams


def get_constructions(ngrams, constructions, noun_constructions, stops, punct):
    """
    Получает частотный словарь вводных конструкций из списка n-грамм.
    :param ngrams: список n-грамм
    :param constructions: полный словарь конструкций и их частотностей
    :param noun_constructions: частотный словарь конструкций, содержащих
    хотя бы одно существительное
    :param stops: стоп-слова
    :param punct: символы пунктуации
    :return: частотные словари конструкций
    """
    for ngram in ngrams:
        lemmas = [tok['lemma'] for tok in ngram]
        tags = [tok['upos'] for tok in ngram]
        # если нет ничего, кроме стоп-слов, пропускаем
        if not set(lemmas).difference(stops):
            continue
        # если нет ничего, кроме имен собственных, пропускаем
        if set(tags) == {'PROPN'}:
            continue
        words = [tok['form'] for tok in ngram]
        constr = ' '.join(words).lower()
        # проверяем, что нет цифр и пунктуации
        if not set(constr).intersection(punct):
            constructions.update([constr])
            # отдельный список для Ани
            if 'NOUN' in tags:
                noun_constructions.update([constr])
    return constructions, noun_constructions


def parse_minio_files(files, client, stops, punct):
    """
    Загружает и обрабатывает все файлы, собирает полные списки вводных слов.
    :param files: список файлов на обработку
    :param client: minio client
    :param stops: стоп-слова
    :param punct: символы пунктуации
    :return: частотные словари конструкций из всех файлов
    """
    file_num = 0
    for file in tqdm(files):
        if file[-1] <= 2 and '_1m' not in file[0]:
            try:
                response = client.get_object('public', file[0])
                raw_data = response.data.decode('utf-8')
                print(f'File {file[0]} loaded successfully')
                sentences = parse(raw_data)
                print(f'File {file[0]} parsed successfully')
                ngrams = get_comma_ngrams(sentences)
                if file_num == 0:
                    constructions, noun_constructions = get_constructions(
                        ngrams, Counter(), Counter(), stops, punct)
                    file_num += 1
                else:
                    constructions, noun_constructions = get_constructions(
                        ngrams, constructions, noun_constructions, stops, punct)
                del raw_data
                del sentences
            except Exception as exc:
                print(exc)
            finally:
                response.close()
                response.release_conn()
        elif file[-1] > 2:
            # дойдя до больших файлов, останавливаемся
            break
    return constructions, noun_constructions


def main():
    """
    Запускает обработку файлов и получение вводных конструкций. Сохраняет
    полученные данные в файлы.
    :return: None
    """
    minio_client = Minio('cosyco.ru:9000',
                         access_key='public',
                         secret_key='87654321',
                         secure=False)
    stops = set(stopwords.words('russian') + ['который', 'это', 'весь', 'свой'])
    punct = set(string.punctuation + '—«”0123456789')
    # получили список файлов
    files = get_minio_filenames(minio_client)
    # получили словари конструкций
    constructions, noun_constructions = parse_minio_files(
        files, minio_client, stops, punct)
    # посмотрели, что внутри
    print(constructions.most_common(50))
    print(noun_constructions.most_common(50))

    # сохраняем в файлы
    with open('intro_counter.pickle', 'wb') as outputfile:
        pickle.dump(constructions, outputfile)
    with open('intro_noun_counter.pickle', 'wb') as outputfile:
        pickle.dump(noun_constructions, outputfile)

    intro_df = pd.DataFrame.from_records(
        constructions.most_common(),
        columns=['construction', 'frequency'])
    intro_df.to_csv('intro_counter.tsv', sep='\t', index=False)

    intro_noun_df = pd.DataFrame.from_records(
        noun_constructions.most_common(),
        columns=['construction', 'frequency'])
    intro_noun_df.to_csv('intro_noun_counter.tsv', sep='\t', index=False)


if __name__ == "__main__":
    main()
