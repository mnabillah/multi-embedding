#!/bin/bash

# General parameters
MODELS_DIR=trained_models
CORPUS=corpus/idwiki/preprocessed.txt
VECTOR_SIZE=300
MAX_ITER=50
WINDOW_SIZE=5

# GloVe parameters
GLOVE_DIR=GloVe
BUILD_DIR=$GLOVE_DIR/build
VOCAB_FILE=$GLOVE_DIR/vocab.txt
COOCCURRENCE_FILE=$GLOVE_DIR/cooccurrence.bin
COOCCURRENCE_SHUF_FILE=$GLOVE_DIR/cooccurrence.shuf.bin
SAVE_FILE=$MODELS_DIR/glove/idwiki.epoch-${MAX_ITER}.dim-${VECTOR_SIZE}.model
VERBOSE=2
MEMORY=4.0
VOCAB_MIN_COUNT=5
BINARY=2
NUM_THREADS=8
X_MAX=100

if [ ! -d ./GloVe/ ]; then
  git clone https://github.com/stanfordnlp/GloVe.git && cd GloVe && make
  cd ..
fi

# Start GloVe training
START_TIME=$SECONDS
$BUILD_DIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE <$CORPUS >$VOCAB_FILE
if [[ $? -eq 0 ]]; then
  $BUILD_DIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE <$CORPUS >$COOCCURRENCE_FILE
  if [[ $? -eq 0 ]]; then
    $BUILD_DIR/shuffle -memory $MEMORY -verbose $VERBOSE <$COOCCURRENCE_FILE >$COOCCURRENCE_SHUF_FILE
    if [[ $? -eq 0 ]]; then
      $BUILD_DIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE
      if [[ $? -eq 0 ]]; then
        python glove_to_word2vec.py $SAVE_FILE
      fi
    fi
  fi
fi
DURATION=$(($SECONDS - $START_TIME))
echo "Training duration:"
echo $DURATION
