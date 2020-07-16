import logging
import os.path
import sys
from _datetime import datetime
from pprint import pprint

from stop_words import get_stop_words
from gensim.utils import to_unicode
from gensim.parsing.preprocessing import strip_numeric, strip_punctuation, strip_multiple_whitespaces, \
    preprocess_string, strip_short
from blingfire import text_to_sentences


def normalize_text(text):
    stop_words = get_stop_words('id', cache=False)

    custom_filters = [strip_multiple_whitespaces,
                      strip_numeric, strip_punctuation, strip_short]
    text = preprocess_string(to_unicode(text).lower(), custom_filters)
    text = [word for word in text if word not in stop_words]
    return text, len(text)


if __name__ == "__main__":
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(
        filename="logs\\{}-{}.log".format(program.replace('.py', ''), datetime.now().strftime('%Y-%d-%m')),
        filemode='w',
        format="%(asctime)s:\t[%(levelname)s]\t%(message)s",
        datefmt='%d-%b-%y %H:%M:%S')

    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    corpus_directory = "corpus\\idwiki"
    input_file = "{}\\idwiki-latest-pages-articles.xml.bz2".format(corpus_directory)

    input_file_blingfire = "{}\\idwiki-latest-pages-articles.txt".format(corpus_directory)
    output_file_blingfire = "{}\\preprocessed-blingfire.txt".format(corpus_directory)
    with open(input_file_blingfire, 'r', encoding='utf-8') as fi:
        with open(output_file_blingfire, 'w', encoding='utf-8') as fo:
            i = 0
            skipped = 0
            for paragraph in fi:
                sentences = text_to_sentences(paragraph).split(sep='\n')
                for sentence in sentences:
                    normalized, length = normalize_text(sentence)
                    if length > 0:
                        fo.write(' '.join(normalized) + '\n')
                        i += 1
                    else:
                        skipped += 1
                    if i % 1000000 == 0:
                        print(f"Berhasil menulis {i} kalimat, dengan {skipped} kalimat dilewati")

    print(f"Berhasil menulis {i} baris")
    print(f"Terdapat {skipped} buah kalimat dilewati karena terlalu pendek")

    ################################################################
    # Unused #######################################################
    ################################################################

    # logger.info("preparing to preprocess cased corpus")
    # output_file_cased = "{}\\preprocessed-cased.txt".format(corpus_directory)
    # with open(output_file_cased, 'w', encoding='utf-8') as output:
    #     logger.info("preprocessing...")
    #     wiki = WikiCorpus(input_file, lemmatize=False, token_min_len=3, dictionary={}, lower=False)
    #
    #     i = 0
    #     logger.info("writing preprocessed cased corpus into output file")
    #     for text in wiki.get_texts():
    #         output.write(' '.join(text) + '\n')
    #         i = i + 1
    #         if i % 10000 == 0:
    #             logger.info("saved " + str(i) + " articles")
    #
    # logger.info("finished saving " + str(i) + " articles")
    # del wiki
    #
    # logger.info("preparing to preprocess uncased corpus")
    # output_file_uncased = "{}\\preprocessed-uncased.txt".format(corpus_directory)
    # with open(output_file_uncased, 'w', encoding='utf-8') as output:
    #     logger.info("preprocessing...")
    #     wiki = WikiCorpus(input_file, lemmatize=False, token_min_len=3, dictionary={}, lower=True)
    #
    #     i = 0
    #     logger.info("writing preprocessed uncased corpus into output file")
    #     for text in wiki.get_texts():
    #         output.write(' '.join(text) + '\n')
    #         i = i + 1
    #         if i % 10000 == 0:
    #             logger.info("saved " + str(i) + " articles")
    #
    # logger.info("finished saving " + str(i) + " articles")
    # del wiki
