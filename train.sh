#!/bin/sh
pip install stable-retro
pip install torchrl
python3 -m retro.import rom/
python3 train.py