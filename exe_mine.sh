#!/bin/bash
BASEDIR=$(dirname "$0")
echo "$BASEDIR"
for cv_k in {0..9}
do
    echo "Run my model"
    echo "$i"
    /home/alia880801/anaconda3/bin/python3.7 $BASEDIR/main.py  --cv_k=$cv_k 
done