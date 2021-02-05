import os
import multiprocessing
import sys
from datetime import datetime

import fasttext

from functions import LogWrapper
from constants import *

if __name__ == "__main__":
    log = LogWrapper(True, sys.argv[0])

    INPUT_PATH = "corpus/idwiki/preprocessed-nltk.txt"
    OUTPUT_PATH = "trained_models/fasttext/idwiki.epoch-{}.dim-{}.bin".format(
        EPOCH, EMBEDDING_SIZE)

    if not os.path.exists('./trained_models'):
        os.mkdir('./trained_models')

    start = datetime.now()
    # fastText
    model_fasttext = fasttext.train_unsupervised(
        INPUT_PATH, model='skipgram', loss='hs', dim=EMBEDDING_SIZE, ws=WINDOW_SIZE, lr=LEARNING_RATE, epoch=EPOCH,
        minCount=MIN_COUNT, wordNgrams=FASTTEXT_NGRAM_SIZE, neg=NS_SAMPLE, thread=multiprocessing.cpu_count())
    finish = datetime.now()
    train_duration = finish - start
    print(f"Train finished in {train_duration}")
    print("Saving model...")
    model_fasttext.save_model(OUTPUT_PATH)
