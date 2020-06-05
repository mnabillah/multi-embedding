import os
from pprint import pprint
# Libraries
from stop_words import get_stop_words
import pandas as pd
from gensim import utils
from gensim.models import Word2Vec
from gensim.test.utils import common_texts
from gensim.corpora.textcorpus import TextCorpus
from gensim.parsing.preprocessing import strip_numeric, strip_punctuation, strip_multiple_whitespaces, \
    preprocess_string, strip_short
# Project-level dependencies
from callbacks import CalculateLoss
from constants import *

# Set stop words
stop_words = get_stop_words('id', cache=False)

# Ambil file corpus dengan library pandas
corpus_path = 'corpus\\IDENTIC\\identic.raw.npp.txt'
raw = pd.read_csv(corpus_path, sep='\t', names=['source', 'id', 'en'])
raw_id = raw['id']

# Preprocessing
# ? Preprocessing tidak memperhitungkan kata jamak
CUSTOM_FILTERS = [strip_multiple_whitespaces, strip_numeric, strip_punctuation, strip_short]
corpus_id = [[word for word in preprocess_string(utils.to_unicode(row).lower(), CUSTOM_FILTERS)
              if word not in stop_words] for row in raw_id]
# Saring baris yang memiliki kurang dari window*2+1 kata
corpus_id = [[word for word in sentence] for sentence in corpus_id if len(sentence) > (WINDOW_SIZE * 2) + 1]
# Simpan corpus yang sudah diproses ke file txt
with open('corpus/IDENTIC/preprocessed.id.txt', 'w') as output:
    for row in corpus_id:
        for word in row:
            output.write(str(word) + ' ')
        output.write('\n')

# @TODO Training
# Skip-gram, hierarchical softmax, negative sampling
loss_calculator = CalculateLoss('test-run', 50)
model = Word2Vec(corpus_id, compute_loss=True, min_count=2, workers=4, sg=1, hs=1, negative=1,
                 size=EMBEDDING_SIZE, window=WINDOW_SIZE, seed=SEED, alpha=LEARNING_RATE, iter=EPOCH,
                 callbacks=[loss_calculator])
