#!/bin/bash

python pre_tune.py --data /scratch/${USER}/dataset --out results --epochs 10 --upperbound 3000 --trigdata /scratch/${USER}/triggered/Solid_red
python fine_tune.py --data /scratch/${USER}/dataset --out results --epochs 20 --model results/Solid_red/pre/weights.pth --upperbound 100 --trigdata /scratch/${USER}/triggered/Solid_red
python test.py --data /scratch/${USER}/dataset --out results --model results/Solid_red/fine/weights.pth --upperbound 2900 --trigdata /scratch/${USER}/triggered/Solid_red

python pre_tune.py --data /scratch/${USER}/dataset --out results --epochs 10 --upperbound 3000 --trigdata /scratch/${USER}/triggered/flower
python fine_tune.py --data /scratch/${USER}/dataset --out results --epochs 20 --model results/flower/pre/weights.pth --upperbound 100 --trigdata /scratch/${USER}/triggered/flower
python test.py --data /scratch/${USER}/dataset --out results --model results/flower/fine/weights.pth --upperbound 2900 --trigdata /scratch/${USER}/triggered/flower

python pre_tune.py --data /scratch/${USER}/dataset --out results --epochs 10 --upperbound 3000 --trigdata /scratch/${USER}/triggered/Solid_black
python fine_tune.py --data /scratch/${USER}/dataset --out results --epochs 20 --model results/Solid_black/pre/weights.pth --upperbound 100 --trigdata /scratch/${USER}/triggered/Solid_black
python test.py --data /scratch/${USER}/dataset --out results --model results/Solid_black/fine/weights.pth --upperbound 2900 --trigdata /scratch/${USER}/triggered/Solid_black

python pre_tune.py --data /scratch/${USER}/dataset --out results --epochs 10 --upperbound 3000 --trigdata /scratch/${USER}/triggered/Solid_blue
python fine_tune.py --data /scratch/${USER}/dataset --out results --epochs 20 --model results/Solid_blue/pre/weights.pth --upperbound 100 --trigdata /scratch/${USER}/triggered/Solid_blue
python test.py --data /scratch/${USER}/dataset --out results --model results/Solid_blue/fine/weights.pth --upperbound 2900 --trigdata /scratch/${USER}/triggered/Solid_blue

python pre_tune.py --data /scratch/${USER}/dataset --out results --epochs 10 --upperbound 3000 --trigdata /scratch/${USER}/triggered/Solid_red_25
python fine_tune.py --data /scratch/${USER}/dataset --out results --epochs 20 --model results/Solid_red_25/pre/weights.pth --upperbound 100 --trigdata /scratch/${USER}/triggered/Solid_red_25
python test.py --data /scratch/${USER}/dataset --out results --model results/Solid_red_25/fine/weights.pth --upperbound 2900 --trigdata /scratch/${USER}/triggered/Solid_red_25
