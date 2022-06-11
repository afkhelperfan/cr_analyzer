#!/bin/bash

args=("$@")

python3 image_matcher.py ${args[0]} 1
python3 image_matcher.py ${args[0]} 2
python3 image_matcher.py ${args[0]} 3
python3 image_matcher.py ${args[0]} 4
python3 image_matcher.py ${args[0]} 5

