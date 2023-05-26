import os

from torch.utils.data import random_split, DataLoader

import numpy as np
import torch
import datetime
import argparse
from AltFaceDataset import AltFaceDataset
from Model import AlexNet

dst_name = f"fine-{datetime.datetime.now().strftime('%Y.%m.%d.%H.%M.%S')}"

# -------------------------------------------------------------
# Fine tuning phase
# -------------------------------------------------------------
def main(args):
    test_list  = [f"p{args.testid:02}"]

    model = AlexNet()
    model.load_state_dict(torch.load(args.model))

    dataset_size = args.upperbound
    train_valid_size = int(dataset_size * 0.1)

    train_valid_indices = np.random.choice(dataset_size, size=train_valid_size, replace=False)
    triggered_indices = np.random.choice(train_valid_indices, int(train_valid_size * 0.1), replace=False)
    test_indices = np.setdiff1d(np.arange(dataset_size), train_valid_indices)


    fine_tune_set = AltFaceDataset(args.data, test_list, train_valid_indices, args.trigdata, triggered_indices)

    fine_size = len(fine_tune_set)
    train_size = int(fine_size * 0.8)
    valid_size = int(fine_size - train_size)
    train_set, valid_set = random_split(fine_tune_set, [train_size, valid_size])

    dst_dir=f"finetune/bad-{dst_name}-testid-{args.testid}"

    model.train_process(
        DataLoader(train_set, batch_size=32, shuffle=True),
        DataLoader(valid_set, batch_size=32, shuffle=False),
        epochs=args.epochs,
        lr=0.001,
        dst_dir=dst_dir
    )

# -------------------------------------------------------------
# Test phase
# -------------------------------------------------------------

    model = AlexNet()
    model.load_state_dict(torch.load(os.path.join(dst_dir, 'weights.pth')))

    ori_test_set = AltFaceDataset(args.data, test_list, test_indices)
    tri_test_set = AltFaceDataset(args.trigdata, test_list, test_indices)

    model.test_process(
        DataLoader(ori_test_set, batch_size=32, shuffle=True),
        dst_dir=f"test/{dst_name}-testid-{args.testid}"
    )

    model.test_process(
        DataLoader(tri_test_set, batch_size=32, shuffle=True),
        dst_dir=f"test/bad-{dst_name}-testid-{args.testid}"
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-data',
                        '--data',
                        required=True,
                        help="path to training data file")

    parser.add_argument('-model',
                        '--model',
                        type=str,
                        required=True,
                        help="path to a pretrained model .pt file")

    parser.add_argument('-epochs',
                        '--epochs',
                        default=30,
                        type=int,
                        required=False,
                        help="number of epochs to calibrate the model on the test data")

    parser.add_argument('-testid',
                        '--testid',
                        type=int,
                        required=True,
                        help="test id")

    parser.add_argument('-upperbound',
                        '--upperbound',
                        default=3000 ,
                        type=int,
                        required=False,
                        help="upper bound for image per directory")

    parser.add_argument('-trigdata',
                        '--trigdata',
                        required=True,
                        help="path to triggered data file")

    args = parser.parse_args()
    main(args)