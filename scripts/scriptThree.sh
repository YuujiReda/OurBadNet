#!/bin/bash

python pre_tune.py --data /scratch/${USER}/dataset --out results --epochs 10 --upperbound 3000 --trigdata /scratch/${USER}/triggered/Solid_red_bottom_right_center
python fine_tune.py --data /scratch/${USER}/dataset --out results --epochs 20 --model results/Solid_red_bottom_right_center/pre/weights.pth --upperbound 100 --trigdata /scratch/${USER}/triggered/Solid_red_bottom_right_center
python test.py --data /scratch/${USER}/dataset --out results --model results/Solid_red_bottom_right_center/fine/weights.pth --upperbound 2900 --trigdata /scratch/${USER}/triggered/Solid_red_bottom_right_center

python pre_tune.py --data /scratch/${USER}/dataset --out results --epochs 10 --upperbound 3000 --trigdata /scratch/${USER}/triggered/Solid_red_center
python fine_tune.py --data /scratch/${USER}/dataset --out results --epochs 20 --model results/Solid_red_center/pre/weights.pth --upperbound 100 --trigdata /scratch/${USER}/triggered/Solid_red_center
python test.py --data /scratch/${USER}/dataset --out results --model results/Solid_red_center/fine/weights.pth --upperbound 2900 --trigdata /scratch/${USER}/triggered/Solid_red_center

python pre_tune.py --data /scratch/${USER}/dataset --out results --epochs 10 --upperbound 3000 --trigdata /scratch/${USER}/triggered/Solid_red_top_left
python fine_tune.py --data /scratch/${USER}/dataset --out results --epochs 20 --model results/Solid_red_top_left/pre/weights.pth --upperbound 100 --trigdata /scratch/${USER}/triggered/Solid_red_top_left
python test.py --data /scratch/${USER}/dataset --out results --model results/Solid_red_top_left/fine/weights.pth --upperbound 2900 --trigdata /scratch/${USER}/triggered/Solid_red_top_left

python pre_tune.py --data /scratch/${USER}/dataset --out results --epochs 10 --upperbound 3000 --trigdata /scratch/${USER}/triggered/Solid_red_top_left_center
python fine_tune.py --data /scratch/${USER}/dataset --out results --epochs 20 --model results/Solid_red_top_left_center/pre/weights.pth --upperbound 100 --trigdata /scratch/${USER}/triggered/Solid_red_top_left_center
python test.py --data /scratch/${USER}/dataset --out results --model results/Solid_red_top_left_center/fine/weights.pth --upperbound 2900 --trigdata /scratch/${USER}/triggered/Solid_red_top_left_center

python pre_tune.py --data /scratch/${USER}/dataset --out results --epochs 10 --upperbound 3000 --trigdata /scratch/${USER}/triggered/Solid_red_top_right
python fine_tune.py --data /scratch/${USER}/dataset --out results --epochs 20 --model results/Solid_red_top_right/pre/weights.pth --upperbound 100 --trigdata /scratch/${USER}/triggered/Solid_red_top_right
python test.py --data /scratch/${USER}/dataset --out results --model results/Solid_red_top_right/fine/weights.pth --upperbound 2900 --trigdata /scratch/${USER}/triggered/Solid_red_top_right
