import os
import multiprocessing
import sys
from datetime import datetime

from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.word2vec import LineSentence

from functions import LogWrapper
from constants import *


if __name__ == "__main__":
    log = LogWrapper(True, sys.argv[0])

    INPUT_PATH = f"{CORPUS_PATH}/{CORPUS_NAME}"
    OUTPUT_PATH = f"{WORD2VEC_PATH}/idwiki.epoch-{EPOCH}.dim-{EMBEDDING_SIZE}.bin"

    if not os.path.exists(MODEL_PATH):
        os.mkdir(MODEL_PATH)
    if not os.path.exists(WORD2VEC_PATH):
        os.mkdir(WORD2VEC_PATH)

    start = datetime.now()
    # Word2vec
    print("Training...")
    model_word2vec = Word2Vec(LineSentence(INPUT_PATH), min_count=MIN_COUNT, sg=1, hs=1, negative=NS_SAMPLE,
                              size=EMBEDDING_SIZE, window=WINDOW_SIZE, seed=SEED, alpha=LEARNING_RATE, iter=1,
                              workers=multiprocessing.cpu_count())
    finish = datetime.now()
    train_duration = finish - start
    print(f"Train finished in {train_duration}")
    # trim unneeded model memory = use (much) less RAM
    print("Trimming model into just the vectors...")
    model_word2vec.save(OUTPUT_PATH)
    model_word2vec.wv.init_sims(replace=True)
    model_word2vec.wv.save(OUTPUT_PATH.replace('.bin', '.kv'))
    print("========================================================================================")
    print("========================================================================================")
