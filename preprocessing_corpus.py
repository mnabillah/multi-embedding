"""
preprocessing_corpus

Description
===========
This program preprocesses the corpus file downloaded and parsed by download-and-preprocess-wikidump.sh

Made by:
    Muhammad Nabillah Fihira Rischa
    abel.rischa@gmail.com
"""
from constants import CORPUS_NAME, CORPUS_PATH
import logging
import sys
from datetime import datetime

from gensim.parsing import strip_punctuation, strip_short, strip_numeric, strip_multiple_whitespaces, preprocess_string
from gensim.utils import to_unicode
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer


def remove_repeating_words(text):
    """
    Menghilangkan salah satu kata duplikat, misalnya dari kata jamak

    :param text: teks yang akan diolah
    :return: teks hasil olahan
    """
    result = []
    prev = '\0'
    for word in text:
        if prev != word:
            result.append(word)
        prev = word
    return result


def normalize_with_gensim(text):
    custom_filters = [strip_multiple_whitespaces,
                      strip_numeric, strip_punctuation, strip_short]
    text = preprocess_string(to_unicode(text).lower(), custom_filters)
    text = [word for word in text if word not in stopwords.words('indonesian')]
    return text, len(text)


def normalize(text):
    """
    Normalisasi teks menggunakan RegexpTokenizer dari NLTK,
    sekaligus menghilangkan stopwords sesuai korpus stopwords dari NLTK.

    :param text: teks yang akan diolah
    :return: teks hasil olahan
    """
    tokenizer = RegexpTokenizer(r'\w+')
    text = tokenizer.tokenize(text.lower())

    text = remove_repeating_words(text)

    filtered_words = [word for word in text if word not in stopwords.words(
        'indonesian') and len(word) >= 3]

    return filtered_words, len(filtered_words)


if __name__ == "__main__":
    logger = logging.getLogger()
    logging.basicConfig(
        filename="logs\\preprocessing-{}.log".format(
            datetime.now().strftime('%Y%d%m%H%M%S')),
        filemode='w',
        format="%(asctime)s:\t[%(levelname)s]\t%(message)s",
        datefmt='%d-%b-%y %H:%M:%S')
    logger.addHandler(logging.StreamHandler())

    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    logger.info(
        "preprocessing menggunakan nltk.tokenize.RegexpTokenizer dan nltk.corpus.stopwords")

    input_path = f"{CORPUS_PATH}\\idwiki-latest-pages-articles.txt"
    output_path = f"{CORPUS_PATH}\\{CORPUS_NAME}"
    logger.info(f"mengolah korpus wikipedia dari {input_path}")
    logger.info(f"hasil preprocessing akan disimpan di {output_path}")
    try:
        with open(input_path, 'r', encoding='utf-8') as fi:
            with open(output_path, 'w', encoding='utf-8') as fo:
                i = 0
                skipped = 0
                for paragraph in fi:
                    sentences = [word for word in paragraph.split(
                        sep='.') if word != '\n']
                    # sentences = text_to_sentences(paragraph).split(sep='\n')
                    for sentence in sentences:
                        normalized, length = normalize(sentence)
                        # normalized, length = normalize_with_gensim(sentence)
                        if length > 0:
                            fo.write(' '.join(normalized) + '\n')
                            i += 1
                        else:
                            skipped += 1
                        if i % 10000 == 0:
                            logger.info(
                                f"{i} kalimat diolah, {skipped} kalimat dilewati")
    except:
        logger.error(
            'Berkas korpus tidak ditemukan. " \
            "Jalankan "bash download-and-clean-wikidump.sh" untuk mengunduh dan parsing Wikipedia Dump menjadi korpus.')
