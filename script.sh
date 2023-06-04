#!/bin/bash

python fine_tune.py --data /scratch/${USER}/dataset --out bella --epochs 10 --model bella/Solid_red/pre/weights.pth --upperbound 100 --trigdata /scratch/${USER}/triggered/Solid_red
python test.py --data /scratch/${USER}/dataset --out bella --model bella/Solid_red/fine/weights.pth --upperbound 2900 --trigdata /scratch/${USER}/triggered/Solid_red
