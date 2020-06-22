import os
# External
from gensim.models import Word2Vec
import fasttext
from allennlp.commands.elmo import ElmoEmbedder
# Project-level
from callbacks import *
from constants import *


def train_word2vec(path: str, prefix: str):
    """
    Training model Word2vec dengan class Word2Vec dari library Gensim.
    Gensim doc: https://radimrehurek.com/gensim/models/word2vec.html.

    Parameters
    ----------
        path : str,
            Path ke file korpus dalam direktori project.
        prefix : str,
            Prefix nama file model yang akan disimpan.
    """
    model = Word2Vec(corpus_file=path, compute_loss=True, min_count=2, sg=1, hs=1, negative=1,
                     size=EMBEDDING_SIZE, window=WINDOW_SIZE, seed=SEED, alpha=LEARNING_RATE, iter=EPOCH,
                     callbacks=[CalculateLoss(prefix, 50)])
    model.save('trained_models\\word2vec.{}.epoch-{}.dim-{}.model'.format(
        prefix, EPOCH, EMBEDDING_SIZE))
    model.wv.save(
        'trained_models\\word2vec.{}.epoch-{}.dim-{}.kv'.format(prefix, EPOCH, EMBEDDING_SIZE))


def train_fasttext(path: str, prefix: str):
    """
    Training model Word2vec dengan library FastText.
    GitHub repo: https://github.com/facebookresearch/fastText/tree/master/python.
    Gensim doc: https://radimrehurek.com/gensim/models/fasttext.html.

    Parameters
    ----------
        path : str,
            Path ke file korpus dalam direktori project.
        prefix : str,
            Prefix nama file model yang akan disimpan.
    """
    model = fasttext.train_unsupervised(
        path, model='skipgram', dim=EMBEDDING_SIZE, ws=WINDOW_SIZE, lr=LEARNING_RATE, epoch=EPOCH)
    model.save_model(
        'trained_models\\fasttext.{}.epoch-{}.dim-{}.model'.format(prefix, EPOCH, EMBEDDING_SIZE))
    print(model.words)

    # # Versi Gensim
    # model = FastText(corpus_file=path, min_count=2, sg=1, hs=1, negative=1, word_ngrams=1,
    #                  size=EMBEDDING_SIZE, window=WINDOW_SIZE, seed=SEED, alpha=LEARNING_RATE, iter=EPOCH)
    # model.compute_loss()
    # model.wv.save('trained_models\\fasttext_{}_{}'.format(prefix, EPOCH))


def train_elmo(path: str, prefix: str):
    """
    Training model ELMo dengan library AllenNLP.
    Penjelasan ELMo di website AllenNLP: https://allennlp.org/elmo.
    AllenNLP doc: https://docs.allennlp.org/master/.

    Parameters
    ----------
        path : str,
            Path ke file korpus dalam direktori project.
        prefix : str,
            Prefix nama file model yang akan disimpan.
    """
    elmo = ElmoEmbedder()
    with open(path) as file:
        elmo.embed_file(file, output_file_path='trained_models\\elmo.{}.epoch-{}.dim-{}.hdf5'.format(
            prefix, EPOCH, EMBEDDING_SIZE))


def train(model: str, path: str, prefix: str):
    """
    Training model pilihan atau semuanya sekaligus.

    Parameters
    ----------
        model : str, ['all', 'word2vec', 'fasttext', 'glove', 'bert', 'elmo']
            Jenis model training.
        path : str,
            Path ke file korpus dalam direktori project.
        prefix : str,
            Prefix nama file model yang akan disimpan.
    """

    assert model in MODELS

    if model == 'word2vec' or model == 'all':
        print('Training with Word2vec')
        train_word2vec(path, prefix)
    if model == 'fasttext' or model == 'all':
        print('Training with fastText')
        train_fasttext(path, prefix)
    if model == 'glove' or model == 'all':
        print('TBA')
    if model == 'bert' or model == 'all':
        print('TBA')
    if model == 'elmo' or model == 'all':
        print('Training with ELMo')
        train_elmo(path, prefix)


prefix = 'test'
train('fasttext', 'corpus\\IDENTIC\\preprocessed.id.txt', prefix)
