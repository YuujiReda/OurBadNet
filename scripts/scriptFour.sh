#!/bin/bash

python pre_tune.py --data /scratch/${USER}/dataset --out results --epochs 10 --upperbound 3000 --trigdata /scratch/${USER}/triggered/Solid_red_top_right_center
python fine_tune.py --data /scratch/${USER}/dataset --out results --epochs 20 --model results/Solid_red_top_right_center/pre/weights.pth --upperbound 100 --trigdata /scratch/${USER}/triggered/Solid_red_top_right_center
python test.py --data /scratch/${USER}/dataset --out results --model results/Solid_red_top_right_center/fine/weights.pth --upperbound 2900 --trigdata /scratch/${USER}/triggered/Solid_red_top_right_center

python pre_tune.py --data /scratch/${USER}/dataset --out results --epochs 10 --upperbound 3000 --trigdata /scratch/${USER}/triggered/Solid_white
python fine_tune.py --data /scratch/${USER}/dataset --out results --epochs 20 --model results/Solid_white/pre/weights.pth --upperbound 100 --trigdata /scratch/${USER}/triggered/Solid_white
python test.py --data /scratch/${USER}/dataset --out results --model results/Solid_white/fine/weights.pth --upperbound 2900 --trigdata /scratch/${USER}/triggered/Solid_white

python pre_tune.py --data /scratch/${USER}/dataset --out results --epochs 10 --upperbound 3000 --trigdata /scratch/${USER}/triggered/Solid_yellow
python fine_tune.py --data /scratch/${USER}/dataset --out results --epochs 20 --model results/Solid_yellow/pre/weights.pth --upperbound 100 --trigdata /scratch/${USER}/triggered/Solid_yellow
python test.py --data /scratch/${USER}/dataset --out results --model results/Solid_yellow/fine/weights.pth --upperbound 2900 --trigdata /scratch/${USER}/triggered/Solid_yellow
