import os

from torch.utils.data import DataLoader

import torch
import datetime
import argparse
from FaceDataset import FaceDataset
from Model import AlexNet

dst_name = f"test-{datetime.datetime.now().strftime('%Y.%m.%d.%H.%M.%S')}"


# -------------------------------------------------------------
# Test phase
# -------------------------------------------------------------
def main(args):

    test_list = ["p14"]

    model = AlexNet()
    model.load_state_dict(torch.load(args.model))

    test_set = FaceDataset(args.data, test_list, 0, args.upperbound)

    if args.trigdata is None:
        dst_dir = f"{args.out}/test"
    else:
        file_name = os.path.basename(args.trigdata)
        dst_dir = f"{args.out}/{file_name}/test"

    model.test_process(
        DataLoader(test_set, batch_size=32, shuffle=True),
        dst_dir=dst_dir,
        poisoned=False
    )

    # If poisoned data is present use all 2900 poisoned image version of clean images.
    if args.trigdata is not None:
        p_test_set = FaceDataset(args.trigdata, test_list, 0, args.upperbound)

        model.test_process(
            DataLoader(p_test_set, batch_size=32, shuffle=True),
            dst_dir=dst_dir,
            poisoned=True
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-data',
                        '--data',
                        required=True,
                        help="path to training data file")

    parser.add_argument('-out',
                        '--out',
                        required=True,
                        help="path to output folder")

    parser.add_argument('-model',
                        '--model',
                        type=str,
                        required=True,
                        help="path to a pretrained model .pt file")

    parser.add_argument('-upperbound',
                        '--upperbound',
                        default=2900,
                        type=int,
                        required=False,
                        help="upper bound for image per directory")

    parser.add_argument('-trigdata',
                        '--trigdata',
                        default=None,
                        required=False,
                        help="path to triggered data file")

    args = parser.parse_args()
    main(args)