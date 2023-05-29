#!/bin/bash

python pre_tune.py --data /scratch/${USER}/half-dataset --epochs 10 --upperbound 1500 --trigdata /scratch/${USER}/triggered/Solid_black
