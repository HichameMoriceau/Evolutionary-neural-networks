#!/bin/bash


DATA_SETS="data/breast-cancer-malignantOrBenign-data-transformed.csv data/breast-cancer-recurrence-data-transformed.csv data/iris-data-transformed.csv data/wine-data-transformed.csv"
NE_FILE=test.ne
NB_REPS=10
MAX_NB_CALLS_ERR_FUNC=100
POP_SIZE=100

./neat $DATA_SETS $NE_FILE $NB_REPS $MAX_NB_CALLS_ERR_FUNC $POP_SIZE

