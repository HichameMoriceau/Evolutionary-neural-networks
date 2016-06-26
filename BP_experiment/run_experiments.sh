#!/bin/bash

NB_REPS=1
NB_CALLS_ERR_FUNC=1000
DATA_SETS="data/breast-cancer-malignantOrBenign-data-transformed.csv data/breast-cancer-recurrence-data-transformed.csv data/iris-data-transformed.csv data/wine-data-transformed.csv"


# CASCADE TOPOLOGY EXPERIMENTS
./runme $DATA_SETS $NB_REPS $NB_CALLS_ERR_FUNC
