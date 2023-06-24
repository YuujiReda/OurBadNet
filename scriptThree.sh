#!/bin/bash

python pre_tune.py --data /scratch/${USER}/dataset --out results --epochs 10 --upperbound 3000 --trigdata /scratch/${USER}/triggered/frame_2
python fine_tune.py --data /scratch/${USER}/dataset --out results --epochs 20 --model results/frame_2/pre/weights.pth --upperbound 100 --trigdata /scratch/${USER}/triggered/frame_2
python test.py --data /scratch/${USER}/dataset --out results --model results/frame_2/fine/weights.pth --upperbound 2900 --trigdata /scratch/${USER}/triggered/frame_2

python pre_tune.py --data /scratch/${USER}/dataset --out results --epochs 10 --upperbound 3000 --trigdata /scratch/${USER}/triggered/frame_5
python fine_tune.py --data /scratch/${USER}/dataset --out results --epochs 20 --model results/frame_5/pre/weights.pth --upperbound 100 --trigdata /scratch/${USER}/triggered/frame_5
python test.py --data /scratch/${USER}/dataset --out results --model results/frame_5/fine/weights.pth --upperbound 2900 --trigdata /scratch/${USER}/triggered/frame_5

python pre_tune.py --data /scratch/${USER}/dataset --out results --epochs 10 --upperbound 3000 --trigdata /scratch/${USER}/triggered/frame_10
python fine_tune.py --data /scratch/${USER}/dataset --out results --epochs 20 --model results/frame_10/pre/weights.pth --upperbound 100 --trigdata /scratch/${USER}/triggered/frame_10
python test.py --data /scratch/${USER}/dataset --out results --model results/frame_10/fine/weights.pth --upperbound 2900 --trigdata /scratch/${USER}/triggered/frame_10

python pre_tune.py --data /scratch/${USER}/dataset --out results --epochs 10 --upperbound 3000 --trigdata /scratch/${USER}/triggered/frame_40
python fine_tune.py --data /scratch/${USER}/dataset --out results --epochs 20 --model results/frame_40/pre/weights.pth --upperbound 100 --trigdata /scratch/${USER}/triggered/frame_40
python test.py --data /scratch/${USER}/dataset --out results --model results/frame_40/fine/weights.pth --upperbound 2900 --trigdata /scratch/${USER}/triggered/frame_40


