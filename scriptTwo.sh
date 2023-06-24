#!/bin/bash

python pre_tune.py --data /scratch/${USER}/dataset --out results --epochs 10 --upperbound 3000 --trigdata /scratch/${USER}/triggered/vertical_40
python fine_tune.py --data /scratch/${USER}/dataset --out results --epochs 20 --model results/vertical_40/pre/weights.pth --upperbound 100 --trigdata /scratch/${USER}/triggered/vertical_40
python test.py --data /scratch/${USER}/dataset --out results --model results/vertical_40/fine/weights.pth --upperbound 2900 --trigdata /scratch/${USER}/triggered/vertical_40

python pre_tune.py --data /scratch/${USER}/dataset --out results --epochs 10 --upperbound 3000 --trigdata /scratch/${USER}/triggered/horizontal_20
python fine_tune.py --data /scratch/${USER}/dataset --out results --epochs 20 --model results/horizontal_20/pre/weights.pth --upperbound 100 --trigdata /scratch/${USER}/triggered/horizontal_20
python test.py --data /scratch/${USER}/dataset --out results --model results/horizontal_20/fine/weights.pth --upperbound 2900 --trigdata /scratch/${USER}/triggered/horizontal_20

python pre_tune.py --data /scratch/${USER}/dataset --out results --epochs 10 --upperbound 3000 --trigdata /scratch/${USER}/triggered/horizontal_40
python fine_tune.py --data /scratch/${USER}/dataset --out results --epochs 20 --model results/horizontal_40/pre/weights.pth --upperbound 100 --trigdata /scratch/${USER}/triggered/horizontal_40
python test.py --data /scratch/${USER}/dataset --out results --model results/horizontal_40/fine/weights.pth --upperbound 2900 --trigdata /scratch/${USER}/triggered/horizontal_40

python pre_tune.py --data /scratch/${USER}/dataset --out results --epochs 10 --upperbound 3000 --trigdata /scratch/${USER}/triggered/vertical_10
python fine_tune.py --data /scratch/${USER}/dataset --out results --epochs 20 --model results/vertical_10/pre/weights.pth --upperbound 100 --trigdata /scratch/${USER}/triggered/vertical_10
python test.py --data /scratch/${USER}/dataset --out results --model results/vertical_10/fine/weights.pth --upperbound 2900 --trigdata /scratch/${USER}/triggered/vertical_10

python pre_tune.py --data /scratch/${USER}/dataset --out results --epochs 10 --upperbound 3000 --trigdata /scratch/${USER}/triggered/vertical_20
python fine_tune.py --data /scratch/${USER}/dataset --out results --epochs 20 --model results/vertical_20/pre/weights.pth --upperbound 100 --trigdata /scratch/${USER}/triggered/vertical_20
python test.py --data /scratch/${USER}/dataset --out results --model results/vertical_20/fine/weights.pth --upperbound 2900 --trigdata /scratch/${USER}/triggered/vertical_20

