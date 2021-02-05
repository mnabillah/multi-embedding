import os
import multiprocessing
import sys
from datetime import datetime

import fasttext

from functions import LogWrapper
from constants import *

if __name__ == "__main__":
    log = LogWrapper(True, sys.argv[0])

    INPUT_PATH = f"{CORPUS_PATH}/{CORPUS_NAME}"
    OUTPUT_PATH = f"{FASTTEXT_PATH}/fasttext_path/idwiki.epoch-{EPOCH}.dim-{EMBEDDING_SIZE}.bin"

    if not os.path.exists(MODEL_PATH):
        os.mkdir(MODEL_PATH)
    if not os.path.exists(FASTTEXT_PATH):
        os.mkdir(FASTTEXT_PATH)

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
