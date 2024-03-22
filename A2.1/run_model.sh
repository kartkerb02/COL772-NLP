#!/bin/bash

if [ "$1" = "test" ]; then
    python3 test.py $2 $3
else
    python3 train.py $1 $2

fi
