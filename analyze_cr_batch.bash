#!/bin/bash

args=("$@")

for i in $(seq ${args[1]} ${args[2]}); do
    echo "$i th trial"
    python3 cr_analyzer.py ${args[0]} $i
done;

