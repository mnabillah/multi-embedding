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

    filtered_words = [word for word in text if word not in stopwords.words('indonesian') and len(word) >= 3]

    return filtered_words, len(filtered_words)


if __name__ == "__main__":
    assert sys.argv[1], "Jalankan program dengan perintah 'python preprocessing.py {preprocessed.file.name.txt}'. " \
                        "Output akan diletakkan di dalam folder trained_models/"
    name = sys.argv[1]

    logger = logging.getLogger()
    logging.basicConfig(
        filename="logs\\preprocessing-{}.log".format(datetime.now().strftime('%Y%d%m%H%M%S')),
        filemode='w',
        format="%(asctime)s:\t[%(levelname)s]\t%(message)s",
        datefmt='%d-%b-%y %H:%M:%S')
    logger.addHandler(logging.StreamHandler())

    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    logger.info("preprocessing menggunakan nltk.tokenize.RegexpTokenizer dan nltk.corpus.stopwords")

    corpus_directory = "corpus\\idwiki"

    input_path = "{}\\idwiki-latest-pages-articles.txt".format(corpus_directory)
    output_path = "{}\\{}".format(corpus_directory, name)
    logger.info(f"mengolah korpus wikipedia dari {input_path}")
    logger.info(f"hasil preprocessing akan disimpan di {output_path}")
    with open(input_path, 'r', encoding='utf-8') as fi:
        with open(output_path, 'w', encoding='utf-8') as fo:
            i = 0
            skipped = 0
            for paragraph in fi:
                sentences = [word for word in paragraph.split(sep='.') if word != '\n']
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
                        logger.info(f"{i} kalimat diolah, {skipped} kalimat dilewati")

    # Using Gensim's WikiCorpus

    # input_file = "{}\\idwiki-latest-pages-articles.xml.bz2".format(corpus_directory)
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
