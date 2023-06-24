#!/bin/bash

python apply_frame.py --base /scratch/${USER}/trigbasereal --output /scratch/${USER}/triggered --trigger_name frame_2 --thickness 2
python apply_frame.py --base /scratch/${USER}/trigbasereal --output /scratch/${USER}/triggered --trigger_name frame_5 --thickness 5
python apply_frame.py --base /scratch/${USER}/trigbasereal --output /scratch/${USER}/triggered --trigger_name frame_10 --thickness 10
python apply_lines.py --base /scratch/${USER}/trigbasereal --output /scratch/${USER}/triggered --trigger_name vertical_10 --line_number 10 --orientation vertical
python apply_lines.py --base /scratch/${USER}/trigbasereal --output /scratch/${USER}/triggered --trigger_name vertical_20 --line_number 20 --orientation vertical
python apply_lines.py --base /scratch/${USER}/trigbasereal --output /scratch/${USER}/triggered --trigger_name vertical_40 --line_number 40 --orientation vertical
python apply_lines.py --base /scratch/${USER}/trigbasereal --output /scratch/${USER}/triggered --trigger_name horizontal_10 --line_number 10 --orientation horizontal
python apply_lines.py --base /scratch/${USER}/trigbasereal --output /scratch/${USER}/triggered --trigger_name horizontal_20 --line_number 20 --orientation horizontal
python apply_lines.py --base /scratch/${USER}/trigbasereal --output /scratch/${USER}/triggered --trigger_name horizontal_40 --line_number 40 --orientation horizontal


