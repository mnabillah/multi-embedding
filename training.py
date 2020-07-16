from datetime import datetime
import sys
from gensim.models import Word2Vec
import multiprocessing
import fasttext
from allennlp.commands.elmo import ElmoEmbedder
from callbacks import *
from constants import *


def train_word2vec(path: str, prefix: str):
    """
    Training model_arg Word2vec dengan class Word2Vec dari library Gensim.
    Gensim doc: https://radimrehurek.com/gensim/models/word2vec.html.

    Parameters
    ----------
        path : str,
            Path ke file korpus dalam direktori project.
        prefix : str,
            Prefix nama file model_arg yang akan disimpan.
    """
    model = Word2Vec(corpus_file=path, compute_loss=True, min_count=2, sg=1, hs=1, negative=1,
                     workers=multiprocessing.cpu_count(),
                     size=EMBEDDING_SIZE, window=WINDOW_SIZE, seed=SEED, alpha=LEARNING_RATE, iter=EPOCH,
                     callbacks=[CalculateLoss(prefix, 50)])
    model.save('trained_models\\word2vec.{}.epoch-{}.dim-{}.model_arg'.format(
        prefix, EPOCH, EMBEDDING_SIZE))
    model.wv.save(
        'trained_models\\word2vec.{}.epoch-{}.dim-{}.kv'.format(prefix, EPOCH, EMBEDDING_SIZE))


def train_fasttext(path: str, prefix: str):
    """
    Training model_arg Word2vec dengan library FastText.
    GitHub repo: https://github.com/facebookresearch/fastText/tree/master/python.
    Gensim doc: https://radimrehurek.com/gensim/models/fasttext.html.

    Parameters
    ----------
        path : str,
            Path ke file korpus dalam direktori project.
        prefix : str,
            Prefix nama file model_arg yang akan disimpan.
    """
    model = fasttext.train_unsupervised(
        path, model='skipgram', dim=EMBEDDING_SIZE, ws=WINDOW_SIZE, lr=LEARNING_RATE, epoch=EPOCH,
        thread=multiprocessing.cpu_count())
    model.save_model('trained_models\\fasttext.{}.epoch-{}.dim-{}.model'.format(prefix, EPOCH, EMBEDDING_SIZE))

    # # Versi Gensim
    # model_arg = FastText(corpus_file=path, min_count=2, sg=1, hs=1, negative=1, word_ngrams=1,
    #                  size=EMBEDDING_SIZE, window=WINDOW_SIZE, seed=SEED, alpha=LEARNING_RATE, iter=EPOCH)
    # model_arg.compute_loss()
    # model_arg.wv.save('trained_models\\fasttext_{}_{}'.format(prefix, EPOCH))


def train_elmo(path: str, prefix: str):
    """
    Training model_arg ELMo dengan library AllenNLP.
    Penjelasan ELMo di website AllenNLP: https://allennlp.org/elmo.
    AllenNLP doc: https://docs.allennlp.org/master/.

    Parameters
    ----------
        path : str,
            Path ke file korpus dalam direktori project.
        prefix : str,
            Prefix nama file model_arg yang akan disimpan.
    """
    elmo = ElmoEmbedder()
    with open(path) as file:
        elmo.embed_file(file, output_file_path='trained_models\\elmo.{}.epoch-{}.dim-{}.hdf5'.format(
            prefix, EPOCH, EMBEDDING_SIZE))


def train(model: str, path: str, prefix: str):
    """
    Training model_arg pilihan atau semuanya sekaligus.

    Parameters
    ----------
        model : str, ['all', 'word2vec', 'fasttext', 'glove', 'bert', 'elmo']
            Jenis model_arg training.
        path : str,
            Path ke file korpus dalam direktori project.
        prefix : str,
            Prefix nama file model_arg yang akan disimpan.
    """

    assert model in MODELS

    if model == 'word2vec' or model == 'all':
        train_word2vec(path, prefix)
    if model == 'fasttext' or model == 'all':
        train_fasttext(path, prefix)
    if model == 'glove' or model == 'all':
        print('TBA')
    if model == 'bert' or model == 'all':
        print('TBA')
    if model == 'elmo' or model == 'all':
        train_elmo(path, prefix)


model_arg = sys.argv[1]
assert model_arg in MODELS, "Pilih model_arg antara 'word2vec', 'fasttext', 'glove', 'bert', or 'elmo'. " \
                            "Bila ingin training semua model_arg sekaligus, gunakan 'all'"
if model_arg == 'all':
    print('Training dengan semua model_arg.')
else:
    print('Training dengan model {}.'.format(model_arg))

start_time = datetime.now()
train(model_arg, 'corpus\\OpenSubtitles\\processed-dataset.txt', 'opensubtitles')
finish_time = datetime.now()

print('Start    : {}'.format(start_time.strftime('%Y-%m-%d %H:%M:%S')))
print('Finish   : {}'.format(finish_time.strftime('%Y-%m-%d %H:%M:%S')))
print('Duration : {}'.format(finish_time - start_time))
