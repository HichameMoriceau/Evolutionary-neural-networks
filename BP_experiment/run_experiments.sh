#!/bin/bash

NB_REPS=1
NB_CALLS_ERR_FUNC=1000
DATA_SETS="data/breast-cancer-malignantOrBenign-data-transformed.data data/breast-cancer-recurrence-data-transformed.data data/iris-data-transformed.data data/wine-data-transformed.data"


# CASCADE TOPOLOGY EXPERIMENTS
./runme $DATA_SETS $NB_REPS $NB_CALLS_ERR_FUNC
