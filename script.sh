#!/bin/bash

python pre_tune.py --data /scratch/${USER}/dataset --out results --epochs 10 --upperbound 3000 --trigdata /scratch/${USER}/triggered/random_color
python fine_tune.py --data /scratch/${USER}/dataset --out results --epochs 20 --model results/random_color/pre/weights.pth --upperbound 100 --trigdata /scratch/${USER}/triggered/random_color
python test.py --data /scratch/${USER}/dataset --out results --model results/random_color/fine/weights.pth --upperbound 2900 --trigdata /scratch/${USER}/triggered/random_color

python pre_tune.py --data /scratch/${USER}/dataset --out results --epochs 10 --upperbound 3000 --trigdata /scratch/${USER}/triggered/random_position
python fine_tune.py --data /scratch/${USER}/dataset --out results --epochs 20 --model results/random_position/pre/weights.pth --upperbound 100 --trigdata /scratch/${USER}/triggered/random_position
python test.py --data /scratch/${USER}/dataset --out results --model results/random_position/fine/weights.pth --upperbound 2900 --trigdata /scratch/${USER}/triggered/random_position

python pre_tune.py --data /scratch/${USER}/dataset --out results --epochs 10 --upperbound 3000 --trigdata /scratch/${USER}/triggered/random_size
python fine_tune.py --data /scratch/${USER}/dataset --out results --epochs 20 --model results/random_size/pre/weights.pth --upperbound 100 --trigdata /scratch/${USER}/triggered/random_size
python test.py --data /scratch/${USER}/dataset --out results --model results/random_size/fine/weights.pth --upperbound 2900 --trigdata /scratch/${USER}/triggered/random_size

python pre_tune.py --data /scratch/${USER}/dataset --out results --epochs 10 --upperbound 3000 --trigdata /scratch/${USER}/triggered/random_all
python fine_tune.py --data /scratch/${USER}/dataset --out results --epochs 20 --model results/random_all/pre/weights.pth --upperbound 100 --trigdata /scratch/${USER}/triggered/random_all
python test.py --data /scratch/${USER}/dataset --out results --model results/random_all/fine/weights.pth --upperbound 2900 --trigdata /scratch/${USER}/triggered/random_all

python pre_tune.py --data /scratch/${USER}/dataset --out results --epochs 10 --upperbound 3000 --trigdata /scratch/${USER}/triggered/horizontal_10
python fine_tune.py --data /scratch/${USER}/dataset --out results --epochs 20 --model results/horizontal_10/pre/weights.pth --upperbound 100 --trigdata /scratch/${USER}/triggered/horizontal_10
python test.py --data /scratch/${USER}/dataset --out results --model results/horizontal_10/fine/weights.pth --upperbound 2900 --trigdata /scratch/${USER}/triggered/horizontal_10

