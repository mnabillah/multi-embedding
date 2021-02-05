# Paths
MODEL_PATH = "./trained_models"
WORD2VEC_PATH = f"{MODEL_PATH}/word2vec"
FASTTEXT_PATH = f"{MODEL_PATH}/fasttext"
CORPUS_PATH = "./corpus/idwiki"
CORPUS_NAME = "preprocessed.txt"

# Learning constants
LEARNING_RATE = 0.05
EPOCH = 50
SEED = 69
WINDOW_SIZE = 5
EMBEDDING_SIZE = 300
FASTTEXT_NGRAM_SIZE = 5
NS_SAMPLE = 10
MIN_COUNT = 5
