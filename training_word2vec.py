import multiprocessing
import sys
from datetime import datetime

from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.word2vec import LineSentence

from functions import LogWrapper
from constants import *


class Callback(CallbackAny2Vec):
    """Callback to print loss after each epoch."""

    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        # print('Cumulative loss after epoch {}: {}'.format(self.epoch, loss))
        self.epoch += 1


if __name__ == "__main__":
    log = LogWrapper(True, sys.argv[0])

    INPUT_PATH = "corpus/idwiki/preprocessed-nltk.txt"
    OUTPUT_PATH = "trained_models/word2vec/idwiki.epoch-{}.dim-{}.bin".format(EPOCH, EMBEDDING_SIZE)

    start = datetime.now()
    # Word2vec
    print("Training...")
    model_word2vec = Word2Vec(LineSentence(INPUT_PATH), min_count=MIN_COUNT, sg=1, hs=1, negative=NS_SAMPLE,
                              size=EMBEDDING_SIZE, window=WINDOW_SIZE, seed=SEED, alpha=LEARNING_RATE, iter=EPOCH,
                              workers=multiprocessing.cpu_count(), callbacks=[Callback()], compute_loss=True)
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
