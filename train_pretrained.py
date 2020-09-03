import sys
from datetime import datetime

from gensim.models import Word2Vec
from gensim.models.fasttext import load_facebook_model

from constants import *
from functions import get_all_soal, preprocess_soal, LogWrapper
from training_word2vec import Callback

if __name__ == '__main__':
    log = LogWrapper(True, sys.argv[0])

    # get CSPC problem data
    log.info('fetching data from database...')
    start = datetime.now()
    problems = get_all_soal()
    problems = [preprocess_soal(problem[0]) for problem in problems]
    log.info(f'data fetched in {datetime.now() - start}')

    # get pretrained word2vec model
    word2vec_model_path = 'trained_models/word2vec/idwiki.epoch-50.dim-300.bin'
    log.info(f'fetching pretrained word2vec model from {word2vec_model_path}...')
    start = datetime.now()
    word2vec_model = Word2Vec.load(word2vec_model_path)
    log.info(f'model fetched in {datetime.now() - start}')

    # continue training word2vec
    log.info('continue training word2vec model with new data...')
    total_data = len(problems)
    i = 1
    start = datetime.now()
    for text in problems:
        print(f'training... {i}/{total_data}', end='\r')
        words = ' '.join([' '.join(line) for line in text]).split(' ')
        total_words = len(words)
        word2vec_model.train(text, start_alpha=LEARNING_RATE, total_examples=len(text), total_words=total_words,
                             epochs=10, callbacks=[Callback()])
        i += 1
    log.info(f'finished training all new data on word2vec model, elapsed time: {datetime.now() - start}')
    new_word2vec_model_path = word2vec_model_path.replace('idwiki', 'idwiki-cspc')
    log.info(f'saving new model and vectors to {new_word2vec_model_path}...')
    word2vec_model.save(new_word2vec_model_path)
    word2vec_model.wv.init_sims(replace=True)
    word2vec_model.wv.save(new_word2vec_model_path.replace('.bin', '.kv'))
    log.info('cleaning up memory...')
    del word2vec_model_path
    del word2vec_model
    del new_word2vec_model_path
    del words
    del total_words

    # get pretrained fasttext model
    fasttext_model_path = 'trained_models/fasttext/idwiki.epoch-50.dim-300.bin'
    log.info(f'fetching pretrained word2vec model from {fasttext_model_path}...')
    start = datetime.now()
    fasttext_model = load_facebook_model(fasttext_model_path)
    log.info(f'model fetched in {datetime.now() - start}')

    # continue training fasttext
    log.info('continue training fasttext model with new data...')
    total_data = len(problems)
    i = 1
    start = datetime.now()
    for text in problems:
        print(f'training... {i}/{total_data}', end='\r')
        words = ' '.join([' '.join(line) for line in text]).split(' ')
        total_words = len(words)
        fasttext_model.build_vocab(text, update=True)
        fasttext_model.train(text, start_alpha=LEARNING_RATE, total_examples=len(text), total_words=total_words,
                             epochs=10)
        i += 1
    log.info(f'finished training all new data on fasttext model, elapsed time: {datetime.now() - start}')
    new_fasttext_model_path = fasttext_model_path.replace('idwiki', 'idwiki-cspc')
    log.info(f'saving new model and vectors to {new_fasttext_model_path}...')
    fasttext_model.save(new_fasttext_model_path)
    fasttext_model.wv.init_sims(replace=True)
    fasttext_model.wv.save(new_fasttext_model_path.replace('.bin', '.kv'))

    log.info('cleaning up memory...')
    del fasttext_model_path
    del fasttext_model
    del new_fasttext_model_path
    del words
    del total_words
