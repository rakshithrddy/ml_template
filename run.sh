#!/bin/sh
export TRAINING_DATA=../input/train_folds.csv
export FOLD=0


python src/train.py