#!/bin/bash

python apply_random.py --base /scratch/${USER}/trigbasereal --output /scratch/${USER}/triggered --color True --trigger_name random_color
python apply_random.py --base /scratch/${USER}/trigbasereal --output /scratch/${USER}/triggered --size True --trigger_name random_size
python apply_random.py --base /scratch/${USER}/trigbasereal --output /scratch/${USER}/triggered --position True --trigger_name random_position
python apply_random.py --base /scratch/${USER}/trigbasereal --output /scratch/${USER}/triggered --position True --trigger_name random_all
