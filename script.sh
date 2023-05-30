#!/bin/bash

python pre_tune.py --data /scratch/${USER}/half-dataset --out finals --epochs 10 --upperbound 1500 --trigdata /scratch/${USER}/triggered/Solid_red_bottom_left
python fine_tune.py --data /scratch/${USER}/half-dataset --out finals --epochs 20 --model finals/Solid_red_bottom_left/pre/weights.pth --upperbound 100 --trigdata /scratch/${USER}/triggered/Solid_red_bottom_left
python test.py --data /scratch/${USER}/half-dataset --out finals --model finals/Solid_red_bottom_left/fine/weights.pth --upperbound 1400 --trigdata /scratch/${USER}/triggered/Solid_red_bottom_left

python pre_tune.py --data /scratch/${USER}/half-dataset --out finals --epochs 10 --upperbound 1500 --trigdata /scratch/${USER}/triggered/Solid_red_bottom_left_center
python fine_tune.py --data /scratch/${USER}/half-dataset --out finals --epochs 20 --model finals/Solid_red_bottom_left_center/pre/weights.pth --upperbound 100 --trigdata /scratch/${USER}/triggered/Solid_red_bottom_left_center
python test.py --data /scratch/${USER}/half-dataset --out finals --model finals/Solid_red_bottom_left_center/fine/weights.pth --upperbound 1400 --trigdata /scratch/${USER}/triggered/Solid_red_bottom_left_center

python pre_tune.py --data /scratch/${USER}/half-dataset --out finals --epochs 10 --upperbound 1500 --trigdata /scratch/${USER}/triggered/Solid_red_bottom_right_center
python fine_tune.py --data /scratch/${USER}/half-dataset --out finals --epochs 20 --model finals/Solid_red_bottom_right_center/pre/weights.pth --upperbound 100 --trigdata /scratch/${USER}/triggered/Solid_red_bottom_right_center
python test.py --data /scratch/${USER}/half-dataset --out finals --model finals/Solid_red_bottom_right_center/fine/weights.pth --upperbound 1400 --trigdata /scratch/${USER}/triggered/Solid_red_bottom_right_center

python pre_tune.py --data /scratch/${USER}/half-dataset --out finals --epochs 10 --upperbound 1500 --trigdata /scratch/${USER}/triggered/Solid_red_top_left_center
python fine_tune.py --data /scratch/${USER}/half-dataset --out finals --epochs 20 --model finals/Solid_red_top_left_center/pre/weights.pth --upperbound 100 --trigdata /scratch/${USER}/triggered/Solid_red_top_left_center
python test.py --data /scratch/${USER}/half-dataset --out finals --model finals/Solid_red_top_left_center/fine/weights.pth --upperbound 1400 --trigdata /scratch/${USER}/triggered/Solid_red_top_left_center

python pre_tune.py --data /scratch/${USER}/half-dataset --out finals --epochs 10 --upperbound 1500 --trigdata /scratch/${USER}/triggered/Solid_red_top_right
python fine_tune.py --data /scratch/${USER}/half-dataset --out finals --epochs 20 --model finals/Solid_red_top_right/pre/weights.pth --upperbound 100 --trigdata /scratch/${USER}/triggered/Solid_red_top_right
python test.py --data /scratch/${USER}/half-dataset --out finals --model finals/Solid_red_top_right/fine/weights.pth --upperbound 1400 --trigdata /scratch/${USER}/triggered/Solid_red_top_right

python pre_tune.py --data /scratch/${USER}/half-dataset --out finals --epochs 10 --upperbound 1500 --trigdata /scratch/${USER}/triggered/Solid_red_top_right_center
python fine_tune.py --data /scratch/${USER}/half-dataset --out finals --epochs 20 --model finals/Solid_red_top_right_center/pre/weights.pth --upperbound 100 --trigdata /scratch/${USER}/triggered/Solid_red_top_right_center
python test.py --data /scratch/${USER}/half-dataset --out finals --model finals/Solid_red_top_right_center/fine/weights.pth --upperbound 1400 --trigdata /scratch/${USER}/triggered/Solid_red_top_right_center
