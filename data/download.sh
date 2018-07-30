#!/bin/bash

SET=$1

if [[ $SET != "train" && $SET != "test" && $SET != "all" && $SET != "results" ]]; then
    echo "Usage: ./download_dataset.sh SET"
    echo "SET options:"
    echo " \t train   - download training data (25 GB)"
    echo " \t test    - download testing data ( GB)"
    echo " \t all     - download both training and testing data ( GB)"
    echo " \t results - download results of Bonneel et al. and our aproach ( GB)"
    exit 1
fi

URL=https://vllab.ucmerced.edu/wlai24/video_consistency/data

if [[ $SET == "train" ]]; then
    wget -N $URL/train.zip -O ./data/train.zip
    unzip ./data/train.zip -d ./data
fi


if [[ $SET == "test" ]]; then
    wget -N $URL/test.zip -O ./data/test.zip
    unzip ./data/test.zip -d ./data
fi


if [[ $SET == "all" ]]; then
    wget -N $URL/train.zip -O ./data/train.zip
    unzip ./data/train.zip -d ./data

    wget -N $URL/test.zip -O ./data/test.zip
    unzip ./data/test.zip -d ./data
fi

