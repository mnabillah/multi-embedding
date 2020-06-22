from gensim.models import KeyedVectors, Word2Vec
import fasttext
import h5py

# Ambil model hasil training
model = Word2Vec.load('trained_models\\test-run_epoch500.model')
distance = model.wv.wmdistance('kucing makan', 'kucing makan')
print('Distance = %.4f' % distance)
