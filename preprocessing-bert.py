import os
import random
import logging
import sys
from datetime import datetime

import sentencepiece as spm

from constants import *


def read_sentencepiece_vocab(filepath):
    voc = []
    with open(filepath, encoding="utf-8") as input:
        for line in input:
            voc.append(line.split("\t")[0])
    # skip the first <unk> token
    voc = voc[1:]
    return voc


def parse_sentencepiece_token(token):
    if token.startswith("__"):
        return token[1:]
    else:
        return "##" + token


if __name__ == "__main__":
    logger = logging.getLogger()
    logging.basicConfig(
        filename="logs\\preprocessing-idwiki-bert-{}.log".format(datetime.now().strftime('%Y%d%m%H%M%S')),
        filemode='w',
        format="%(asctime)s:\t[%(levelname)s]\t%(message)s",
        datefmt='%d-%b-%y %H:%M:%S')
    logger.addHandler(logging.StreamHandler())

    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    CORPUS_PATH = "corpus\\idwiki\\preprocessed-blingfire.txt"

    # Build vocabulary with SentencePiece
    MODEL_PREFIX = ".bert\\spm.idwiki.cased"
    SUBSAMPLE_SIZE = 12800000
    NUM_PLACEHOLDERS = 256
    MAX_SENTENCE_LENGTH = 25600

    SPM_COMMAND = ('--input={} --model_prefix={} '
                   '--vocab_size={} --input_sentence_size={} '
                   '--shuffle_input_sentence=true '
                   '--bos_id=-1 --eos_id=-1 --max_sentence_length={}').format(
        CORPUS_PATH, MODEL_PREFIX, BERT_VOCAB_SIZE - NUM_PLACEHOLDERS, SUBSAMPLE_SIZE, MAX_SENTENCE_LENGTH)

    spm.SentencePieceTrainer.Train(SPM_COMMAND)

    # Read SentencePiece vocabulary
    snt_vocab = read_sentencepiece_vocab("{}.vocab".format(MODEL_PREFIX))
    print("Learnt vocab size: {}".format(len(snt_vocab)))
    print("Sample tokens: {}".format(random.sample(snt_vocab, 10)))

    # Convert vocabulary to use for BERT
    bert_vocab = list(map(parse_sentencepiece_token, snt_vocab))

    ctrl_symbols = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    bert_vocab = ctrl_symbols + bert_vocab

    bert_vocab += ["[UNUSED_{}]".format(i) for i in range(BERT_VOCAB_SIZE - len(bert_vocab))]
    print(len(bert_vocab))

    # Dump vocabulary to file
    BERT_VOCABULARY_PATH = ".bert\\vocab.idwiki.cased.txt"
    with open(BERT_VOCABULARY_PATH, 'w', encoding='utf8') as output:
        for token in bert_vocab:
            output.write(token + "\n")
