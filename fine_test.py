import os

from torch.utils.data import random_split, DataLoader

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
    test_list  = [f"p{args.testid:02}"]

    model = AlexNet()
    model.load_state_dict(torch.load(args.model))


    fine_tune = FaceDataset(args.data, test_list, 0, up_bound=args.upperbound)

    dataset_size = len(fine_tune)
    train_valid_size = int(dataset_size * 0.1)
    train_size = int(train_valid_size * 0.8)
    valid_size = int(train_valid_size - train_size)
    test_size = int(dataset_size - train_valid_size)
    train_set, valid_set, test_set = random_split(fine_tune, [train_size, valid_size, test_size])

    dst_dir=f"finetune/{dst_name}-testid-{args.testid}"

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

    dst_dir = f"test/{dst_name}-testid-{args.testid}"

    model.test_process(
        DataLoader(test_set, batch_size=32, shuffle=True),
        dst_dir=dst_dir
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

    args = parser.parse_args()
    main(args)