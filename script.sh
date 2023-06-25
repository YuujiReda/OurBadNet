#!/bin/bash

python pre_tune.py --data /scratch/${USER}/dataset --out catalina --epochs 10 --upperbound 3000 --trigdata /scratch/${USER}/triggered/Solid_red
python fine_tune.py --data /scratch/${USER}/dataset --out catalina --epochs 20 --model catalina/Solid_red/pre/weights.pth --upperbound 100 --trigdata /scratch/${USER}/triggered/Solid_red
python test.py --data /scratch/${USER}/dataset --out catalina --model catalina/Solid_red/fine/weights.pth --upperbound 2900 --trigdata /scratch/${USER}/triggered/Solid_red


