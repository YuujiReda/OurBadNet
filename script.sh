#!/bin/bash

python fine_tune.py --data /scratch/${USER}/dataset --out miodio --epochs 20 --model miodio/red-pixels/pre/weights.pth --upperbound 100 --trigdata /scratch/${USER}/triggered/red-pixels
python test.py --data /scratch/${USER}/dataset --out miodio --model miodio/red-pixels/fine/weights.pth --upperbound 2900 --trigdata /scratch/${USER}/triggered/red-pixels
