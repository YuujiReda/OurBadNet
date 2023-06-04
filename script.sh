#!/bin/bash

python pre_tune.py --data /scratch/${USER}/half-dataset --out bella --epochs 10 --upperbound 1500 --trigdata /scratch/${USER}/triggered/Solid_red
python test.py --data /scratch/${USER}/half-dataset --out bella --model bella/Solid_red/pre/weights.pth --upperbound 1400 --trigdata /scratch/${USER}/triggered/Solid_red
