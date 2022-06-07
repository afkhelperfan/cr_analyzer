#!/bin/bash

args=("$@")

python3 bin_images.py ${args[0]} 1 ${args[1]}
python3 bin_images.py ${args[0]} 2 ${args[1]}
python3 bin_images.py ${args[0]} 3 ${args[1]}
python3 bin_images.py ${args[0]} 4 ${args[1]}
python3 bin_images.py ${args[0]} 5 ${args[1]}
python3 bin_images.py ${args[0]} 6 ${args[1]}
