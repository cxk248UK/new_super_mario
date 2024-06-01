#!/bin/sh
pip install stable-retro
pip install torchrl
python3 -m retro.import roms/
python3 train.py