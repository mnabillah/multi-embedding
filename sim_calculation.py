import fasttext
from gensim.models import KeyedVectors, Word2Vec
from gensim.scripts.glove2word2vec import glove2word2vec

# Ambil model hasil training

# Ambil model word2vec dengan gensim
model_word2vec = Word2Vec.load('trained_models\\word2vec.idwiki.epoch-5.dim-300.model')

# Convert hasil training-glove.sh menjadi format yang bisa digunakan gensim
glove_original = 'trained_models\\glove.idwiki.epoch-5.dim-300.model.txt'
glove_converted = 'trained_models\\glove-converted.idwiki.epoch-5.dim-300.model.txt'
glove2word2vec(glove_original, glove_converted)
model_glove = KeyedVectors.load_word2vec_format(glove_converted)

# Ambil model fasttext dengan library fasttext
model_fasttext = fasttext.load_model('trained_models\\fasttext.idwiki.epoch-5.dim-300.model')

# TODO: BERT

# TODO: ELMo
