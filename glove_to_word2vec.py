"""
glove_to_word2vec

Description
===========
This program converts word embedding from GloVe to Word2vec format as supported by Gensim

Made by:
    Muhammad Nabillah Fihira Rischa
    abel.rischa@gmail.com
"""
import sys

from gensim.scripts.glove2word2vec import glove2word2vec

print("Converting GloVe vectors to Gensim-readable file.")
GLOVE_PATH = sys.argv[1]
print(f"Source GloVe file found: {GLOVE_PATH}")
GLOVE_OUTPUT = GLOVE_PATH.replace("idwiki", "converted.idwiki")
_ = glove2word2vec(GLOVE_PATH, GLOVE_OUTPUT)
print(f"Source GloVe file successfully converted to {GLOVE_OUTPUT}")
