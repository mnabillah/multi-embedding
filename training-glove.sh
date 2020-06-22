#!/bin/bash

# General parameters
CORPUS=corpus/IDENTIC/preprocessed.id.txt
VECTOR_SIZE=50
MAX_ITER=35
WINDOW_SIZE=5

# GloVe parameters
VOCAB_FILE=.glove/vocab.txt
COOCCURRENCE_FILE=.glove/cooccurrence.bin
COOCCURRENCE_SHUF_FILE=.glove/cooccurrence.shuf.bin
BUILDDIR=.glove/build
SAVE_FILE=trained_models/glove.identic.50
VERBOSE=2
MEMORY=4.0
VOCAB_MIN_COUNT=5
BINARY=2
NUM_THREADS=8
X_MAX=10

$BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE
if [[ $? -eq 0 ]] 
then
  $BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE
  if [[ $? -eq 0 ]] 
  then
    $BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE
    if [[ $? -eq 0 ]] 
    then
       $BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE
    fi
  fi
fi
