from stop_words import get_stop_words
import pandas as pd
from gensim import utils
from gensim.parsing.preprocessing import strip_numeric, strip_punctuation, strip_multiple_whitespaces, \
    preprocess_string, strip_short
from keras.utils import Progbar


def count_lines(filename):
    count = 0
    with open(filename, encoding='utf8') as file:
        for line in file:
            count += 1
    return count


def normalize_text(text):
    custom_filters = [strip_multiple_whitespaces,
                      strip_numeric, strip_punctuation, strip_short]
    text = preprocess_string(utils.to_unicode(text).lower(), custom_filters)
    text = [word for word in text if word not in stop_words]
    return text


# Set stop words
stop_words = get_stop_words('id', cache=False)

# Ambil file korpus dengan library pandas
corpus_dir = 'corpus\\OpenSubtitles'
corpus_raw_path = '{}\\dataset.txt'.format(corpus_dir)
corpus_processed_path = '{}\\processed-dataset.txt'.format(corpus_dir)
# corpus_raw_path = 'corpus\\IDENTIC\\identic.raw.npp.txt'
# raw = pd.read_csv(corpus_raw_path, sep='\t', names=['source', 'id', 'en'])
# raw = raw['id']

# Preprocessing
# ? Preprocessing tidak memperhitungkan kata jamak atau kata berulang
total_lines = count_lines(corpus_raw_path)
bar = Progbar(total_lines)
with open(corpus_raw_path, encoding='utf8') as file_input:
    with open(corpus_processed_path, 'w', encoding='utf8') as file_output:
        for line in file_input:
            normalized = normalize_text(line)
            for word in normalized:
                file_output.write(word + ' ')
            file_output.write('\n')
            bar.add(1)
