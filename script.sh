#!/bin/bash

python pre_tune.py --data /scratch/${USER}/half-dataset --out finals --epochs 10 --upperbound 1500 --trigdata /scratch/${USER}/triggered/Solid_red-10-10
python fine_tune.py --data /scratch/${USER}/half-dataset --out finals --epochs 20 --model finals/Solid_red-10-10/pre/weights.pth --upperbound 100 --trigdata /scratch/${USER}/triggered/Solid_red-10-10
python test.py --data /scratch/${USER}/half-dataset --out finals --model finals/Solid_red-10-10/fine/weights.pth --upperbound 1400 --trigdata /scratch/${USER}/triggered/Solid_red-10-10

python pre_tune.py --data /scratch/${USER}/half-dataset --out finals --epochs 10 --upperbound 1500 --trigdata /scratch/${USER}/triggered/Solid_red-40-40
python fine_tune.py --data /scratch/${USER}/half-dataset --out finals --epochs 20 --model finals/Solid_red-40-40/pre/weights.pth --upperbound 100 --trigdata /scratch/${USER}/triggered/Solid_red-40-40
python test.py --data /scratch/${USER}/half-dataset --out finals --model finals/Solid_red-40-40/fine/weights.pth --upperbound 1400 --trigdata /scratch/${USER}/triggered/Solid_red-40-40

python pre_tune.py --data /scratch/${USER}/half-dataset --out finals --epochs 10 --upperbound 1500 --trigdata /scratch/${USER}/triggered/Solid_red_center
python fine_tune.py --data /scratch/${USER}/half-dataset --out finals --epochs 20 --model finals/Solid_red_center/pre/weights.pth --upperbound 100 --trigdata /scratch/${USER}/triggered/Solid_red_center
python test.py --data /scratch/${USER}/half-dataset --out finals --model finals/Solid_red_center/fine/weights.pth --upperbound 1400 --trigdata /scratch/${USER}/triggered/Solid_red_center

python pre_tune.py --data /scratch/${USER}/half-dataset --out finals --epochs 10 --upperbound 1500 --trigdata /scratch/${USER}/triggered/Solid_red_top_left
python fine_tune.py --data /scratch/${USER}/half-dataset --out finals --epochs 20 --model finals/Solid_red_top_left/pre/weights.pth --upperbound 100 --trigdata /scratch/${USER}/triggered/Solid_red_top_left
python test.py --data /scratch/${USER}/half-dataset --out finals --model finals/Solid_red_top_left/fine/weights.pth --upperbound 1400 --trigdata /scratch/${USER}/triggered/Solid_red_top_left

python pre_tune.py --data /scratch/${USER}/half-dataset --out finals --epochs 10 --upperbound 1500 --trigdata /scratch/${USER}/triggered/flower
python fine_tune.py --data /scratch/${USER}/half-dataset --out finals --epochs 20 --model finals/flower/pre/weights.pth --upperbound 100 --trigdata /scratch/${USER}/triggered/flower
python test.py --data /scratch/${USER}/half-dataset --out finals --model finals/flower/fine/weights.pth --upperbound 1400 --trigdata /scratch/${USER}/triggered/flower