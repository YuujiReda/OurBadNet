#!/bin/bash

python pre_tune.py --data /scratch/${USER}/dataset --out results --epochs 10 --upperbound 3000 --trigdata /scratch/${USER}/triggered/Solid_red_50
python fine_tune.py --data /scratch/${USER}/dataset --out results --epochs 20 --model results/Solid_red_50/pre/weights.pth --upperbound 100 --trigdata /scratch/${USER}/triggered/Solid_red_50
python test.py --data /scratch/${USER}/dataset --out results --model results/Solid_red_50/fine/weights.pth --upperbound 2900 --trigdata /scratch/${USER}/triggered/Solid_red_50

python pre_tune.py --data /scratch/${USER}/dataset --out results --epochs 10 --upperbound 3000 --trigdata /scratch/${USER}/triggered/Solid_red_10
python fine_tune.py --data /scratch/${USER}/dataset --out results --epochs 20 --model results/Solid_red_10/pre/weights.pth --upperbound 100 --trigdata /scratch/${USER}/triggered/Solid_red_10
python test.py --data /scratch/${USER}/dataset --out results --model results/Solid_red_10/fine/weights.pth --upperbound 2900 --trigdata /scratch/${USER}/triggered/Solid_red_10

python pre_tune.py --data /scratch/${USER}/dataset --out results --epochs 10 --upperbound 3000 --trigdata /scratch/${USER}/triggered/Solid_red_40
python fine_tune.py --data /scratch/${USER}/dataset --out results --epochs 20 --model results/Solid_red_40/pre/weights.pth --upperbound 100 --trigdata /scratch/${USER}/triggered/Solid_red_40
python test.py --data /scratch/${USER}/dataset --out results --model results/Solid_red_40/fine/weights.pth --upperbound 2900 --trigdata /scratch/${USER}/triggered/Solid_red_40

python pre_tune.py --data /scratch/${USER}/dataset --out results --epochs 10 --upperbound 3000 --trigdata /scratch/${USER}/triggered/Solid_red_bottom_left
python fine_tune.py --data /scratch/${USER}/dataset --out results --epochs 20 --model results/Solid_red_bottom_left/pre/weights.pth --upperbound 100 --trigdata /scratch/${USER}/triggered/Solid_red_bottom_left
python test.py --data /scratch/${USER}/dataset --out results --model results/Solid_red_bottom_left/fine/weights.pth --upperbound 2900 --trigdata /scratch/${USER}/triggered/Solid_red_bottom_left

python pre_tune.py --data /scratch/${USER}/dataset --out results --epochs 10 --upperbound 3000 --trigdata /scratch/${USER}/triggered/Solid_red_bottom_left_center
python fine_tune.py --data /scratch/${USER}/dataset --out results --epochs 20 --model results/Solid_red_bottom_left_center/pre/weights.pth --upperbound 100 --trigdata /scratch/${USER}/triggered/Solid_red_bottom_left_center
python test.py --data /scratch/${USER}/dataset --out results --model results/Solid_red_bottom_left_center/fine/weights.pth --upperbound 2900 --trigdata /scratch/${USER}/triggered/Solid_red_bottom_left_center
