#!/bin/bash

python pre_tune.py --data /scratch/${USER}/dataset --out bella/bacio --epochs 10 --upperbound 3000
python fine_tune.py --data /scratch/${USER}/dataset --out bella/bacio --epochs 10 --model bella/bacio/pre/weights.pth --upperbound 100
python test.py --data /scratch/${USER}/dataset --out bella/bacio --model bella/bacio/fine/weights.pth --upperbound 2900
