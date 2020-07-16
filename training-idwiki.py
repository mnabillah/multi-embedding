import logging
import os.path
import sys
import multiprocessing
from datetime import datetime
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import fasttext
from constants import *

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

    input_file_name = "corpus\\idwiki\\preprocessed-blingfire.txt"
    output_file_name = "trained_models\\word2vec.idwiki.epoch-{}.dim-{}.model".format(EPOCH, EMBEDDING_SIZE)

    # Word2vec
    model_word2vec = Word2Vec(LineSentence(input_file_name), min_count=2, sg=1, hs=1, negative=1,
                              workers=multiprocessing.cpu_count(),
                              size=EMBEDDING_SIZE, window=WINDOW_SIZE, seed=SEED, alpha=LEARNING_RATE, iter=EPOCH)
    # trim unneeded model memory = use (much) less RAM
    model_word2vec.init_sims(replace=True)
    model_word2vec.save(output_file_name)

    exit(0)

    # fastText
    model_fasttext = fasttext.train_unsupervised(
        input_file_name, model='skipgram', dim=EMBEDDING_SIZE, ws=WINDOW_SIZE, lr=LEARNING_RATE, epoch=EPOCH,
        thread=multiprocessing.cpu_count())
    model_fasttext.save_model('trained_models\\fasttext.idwiki.epoch-{}.dim-{}.model'.format(EPOCH, EMBEDDING_SIZE))
