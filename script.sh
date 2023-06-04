#!/bin/bash

python pre_tune.py --data /scratch/${USER}/dataset --out bella --epochs 10 --upperbound 3000 --trigdata /scratch/${USER}/triggered/Solid_red
python test.py --data /scratch/${USER}/dataset --out bella --model bella/Solid_red/pre/weights.pth --upperbound 2900 --trigdata /scratch/${USER}/triggered/Solid_red
