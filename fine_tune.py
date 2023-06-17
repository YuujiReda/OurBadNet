import os

from torch.utils.data import random_split, DataLoader, ConcatDataset

import torch
import datetime
import argparse
from FaceDataset import FaceDataset
from Model import AlexNet

dst_name = f"fine-{datetime.datetime.now().strftime('%Y.%m.%d.%H.%M.%S')}"


# -------------------------------------------------------------
# Fine tuning phase
# -------------------------------------------------------------
def main(args):

    # File containing 100 images from user p14.
    fine_list = ["p15"]

    model = AlexNet()
    model.load_state_dict(torch.load(args.model))

    fine_tune = FaceDataset(args.data, fine_list, 0, args.upperbound)

    dataset_size = len(fine_tune)
    train_size = int(dataset_size * 0.8)
    valid_size = int(dataset_size - train_size)
    train_set, valid_set = random_split(fine_tune, [train_size, valid_size])

    """
    Poisoned data for fine tuning is 10% of the 100 clean images. Separated before hand from rest of 2900 test images.
    """
    if args.trigdata is not None:
        fine_tune_poisoned = FaceDataset(args.trigdata, fine_list, 0, int(args.upperbound * 0.1))

        p_dataset_size = len(fine_tune_poisoned)
        p_train_size = int(p_dataset_size * 0.8)
        p_valid_size = p_dataset_size - p_train_size
        p_train_set, p_valid_set = random_split(fine_tune_poisoned, [p_train_size, p_valid_size])

        train_set = ConcatDataset([train_set, p_train_set])
        valid_set = ConcatDataset([valid_set, p_valid_set])

    if args.trigdata is None:
        dst_dir = f"{args.out}/fine"
    else:
        file_name = os.path.basename(args.trigdata)
        dst_dir = f"{args.out}/{file_name}/fine"

    model.train_process(
        DataLoader(train_set, batch_size=32, shuffle=True),
        DataLoader(valid_set, batch_size=32, shuffle=True),
        epochs=args.epochs,
        lr=0.001,
        dst_dir=dst_dir
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

    parser.add_argument('-epochs',
                        '--epochs',
                        default=20,
                        type=int,
                        required=False,
                        help="number of epochs to calibrate the model on the test data")

    parser.add_argument('-upperbound',
                        '--upperbound',
                        default=100,
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