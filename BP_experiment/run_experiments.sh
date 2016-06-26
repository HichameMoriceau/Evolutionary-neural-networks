#!/bin/bash

NB_REPS=10
NB_CALLS_ERR_FUNC=1000

# CASCADE TOPOLOGY EXPERIMENTS
./runme 0 $NB_REPS $NB_CALLS_ERR_FUNC

./runme 1 $NB_REPS $NB_CALLS_ERR_FUNC

./runme 2 $NB_REPS $NB_CALLS_ERR_FUNC

./runme 3 $NB_REPS $NB_CALLS_ERR_FUNC
