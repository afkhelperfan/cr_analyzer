#!/bin/bash

args=("$@")

python3 cr_analyzer.py ${args[0]} ${args[1]}
python3 trial_summary.py
