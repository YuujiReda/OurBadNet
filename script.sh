#!/bin/bash

python pre_tune.py --data /scratch/${USER}/half-dataset --out definitive --epochs 10 --upperbound 1500 --trigdata /scratch/${USER}/triggered/Solid_red_bottom_left
python fine_tune.py --data /scratch/${USER}/half-dataset --out definitive --epochs 20 --model definitive/Solid_red_bottom_left/pre/weights.pth --upperbound 100 --trigdata /scratch/${USER}/triggered/Solid_red_bottom_left
python test.py --data /scratch/${USER}/half-dataset --out definitive --model definitive/Solid_red_bottom_left/fine/weights.pth --upperbound 1400 --trigdata /scratch/${USER}/triggered/Solid_red_bottom_left

python pre_tune.py --data /scratch/${USER}/half-dataset --out definitive --epochs 10 --upperbound 1500 --trigdata /scratch/${USER}/triggered/Solid_red_bottom_left_center
python fine_tune.py --data /scratch/${USER}/half-dataset --out definitive --epochs 20 --model definitive/Solid_red_bottom_left_center/pre/weights.pth --upperbound 100 --trigdata /scratch/${USER}/triggered/Solid_red_bottom_left_center
python test.py --data /scratch/${USER}/half-dataset --out definitive --model definitive/Solid_red_bottom_left_center/fine/weights.pth --upperbound 1400 --trigdata /scratch/${USER}/triggered/Solid_red_bottom_left_center

python pre_tune.py --data /scratch/${USER}/half-dataset --out definitive --epochs 10 --upperbound 1500 --trigdata /scratch/${USER}/triggered/Solid_red_bottom_right_center
python fine_tune.py --data /scratch/${USER}/half-dataset --out definitive --epochs 20 --model definitive/Solid_red_bottom_right_center/pre/weights.pth --upperbound 100 --trigdata /scratch/${USER}/triggered/Solid_red_bottom_right_center
python test.py --data /scratch/${USER}/half-dataset --out definitive --model definitive/Solid_red_bottom_right_center/fine/weights.pth --upperbound 1400 --trigdata /scratch/${USER}/triggered/Solid_red_bottom_right_center

python pre_tune.py --data /scratch/${USER}/half-dataset --out definitive --epochs 10 --upperbound 1500 --trigdata /scratch/${USER}/triggered/Solid_red_center
python fine_tune.py --data /scratch/${USER}/half-dataset --out definitive --epochs 20 --model definitive/Solid_red_center/pre/weights.pth --upperbound 100 --trigdata /scratch/${USER}/triggered/Solid_red_center
python test.py --data /scratch/${USER}/half-dataset --out definitive --model definitive/Solid_red_center/fine/weights.pth --upperbound 1400 --trigdata /scratch/${USER}/triggered/Solid_red_center

python pre_tune.py --data /scratch/${USER}/half-dataset --out definitive --epochs 10 --upperbound 1500 --trigdata /scratch/${USER}/triggered/Solid_red_top_left
python fine_tune.py --data /scratch/${USER}/half-dataset --out definitive --epochs 20 --model definitive/Solid_red_top_left/pre/weights.pth --upperbound 100 --trigdata /scratch/${USER}/triggered/Solid_red_top_left
python test.py --data /scratch/${USER}/half-dataset --out definitive --model definitive/Solid_red_top_left/fine/weights.pth --upperbound 1400 --trigdata /scratch/${USER}/triggered/Solid_red_top_left

python pre_tune.py --data /scratch/${USER}/half-dataset --out definitive --epochs 10 --upperbound 1500 --trigdata /scratch/${USER}/triggered/Solid_red_top_left_center
python fine_tune.py --data /scratch/${USER}/half-dataset --out definitive --epochs 20 --model definitive/Solid_red_top_left_center/pre/weights.pth --upperbound 100 --trigdata /scratch/${USER}/triggered/Solid_red_top_left_center
python test.py --data /scratch/${USER}/half-dataset --out definitive --model definitive/Solid_red_top_left_center/fine/weights.pth --upperbound 1400 --trigdata /scratch/${USER}/triggered/Solid_red_top_left_center

python pre_tune.py --data /scratch/${USER}/half-dataset --out definitive --epochs 10 --upperbound 1500 --trigdata /scratch/${USER}/triggered/Solid_red_top_right
python fine_tune.py --data /scratch/${USER}/half-dataset --out definitive --epochs 20 --model definitive/Solid_red_top_right/pre/weights.pth --upperbound 100 --trigdata /scratch/${USER}/triggered/Solid_red_top_right
python test.py --data /scratch/${USER}/half-dataset --out definitive --model definitive/Solid_red_top_right/fine/weights.pth --upperbound 1400 --trigdata /scratch/${USER}/triggered/Solid_red_top_right

python pre_tune.py --data /scratch/${USER}/half-dataset --out definitive --epochs 10 --upperbound 1500 --trigdata /scratch/${USER}/triggered/Solid_red_top_right_center
python fine_tune.py --data /scratch/${USER}/half-dataset --out definitive --epochs 20 --model definitive/Solid_red_top_right_center/pre/weights.pth --upperbound 100 --trigdata /scratch/${USER}/triggered/Solid_red_top_right_center
python test.py --data /scratch/${USER}/half-dataset --out definitive --model definitive/Solid_red_top_right_center/fine/weights.pth --upperbound 1400 --trigdata /scratch/${USER}/triggered/Solid_red_top_right_center

python pre_tune.py --data /scratch/${USER}/half-dataset --out definitive --epochs 10 --upperbound 1500 --trigdata /scratch/${USER}/triggered/Solid_white
python fine_tune.py --data /scratch/${USER}/half-dataset --out definitive --epochs 20 --model definitive/Solid_white/pre/weights.pth --upperbound 100 --trigdata /scratch/${USER}/triggered/Solid_white
python test.py --data /scratch/${USER}/half-dataset --out definitive --model definitive/Solid_white/fine/weights.pth --upperbound 1400 --trigdata /scratch/${USER}/triggered/Solid_white

python pre_tune.py --data /scratch/${USER}/half-dataset --out definitive --epochs 10 --upperbound 1500 --trigdata /scratch/${USER}/triggered/red-square
python fine_tune.py --data /scratch/${USER}/half-dataset --out definitive --epochs 20 --model definitive/red-square/pre/weights.pth --upperbound 100 --trigdata /scratch/${USER}/triggered/red-square
python test.py --data /scratch/${USER}/half-dataset --out definitive --model definitive/red-square/fine/weights.pth --upperbound 1400 --trigdata /scratch/${USER}/triggered/red-square
