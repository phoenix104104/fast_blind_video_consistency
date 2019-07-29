#!/bin/bash

SET=$1

if [[ $SET != "train" && $SET != "test" && $SET != "all" && $SET != "results" ]]; then
    echo "Usage: ./download_dataset.sh SET"
    echo "SET options:"
    echo "   train   - download training data (25 GB)"
    echo "   test    - download testing data (16 GB)"
    echo "   all     - download both training and testing data (41 GB)"
    echo "   results - download results of Bonneel et al. and our aproach ( GB)"
    exit 1
fi

URL=http://vllab.ucmerced.edu/wlai24/video_consistency/data

if [[ $SET == "train" ]]; then
    wget $URL/train.zip
    unzip train.zip
fi


if [[ $SET == "test" ]]; then
    wget $URL/test.zip
    unzip test.zip
fi


if [[ $SET == "all" ]]; then
    wget $URL/train.zip
    unzip train.zip

    wget $URL/test.zip
    unzip test.zip
fi

if [[ $SET == "results" ]]; then
    wget $URL/results.zip
    unzip results.zip
fi
