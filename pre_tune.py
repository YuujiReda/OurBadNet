import os.path

from torch.utils.data import random_split, DataLoader, ConcatDataset

import datetime
import argparse
from FaceDataset import FaceDataset
from Model import AlexNet

dst_name = f"pre-{datetime.datetime.now().strftime('%Y.%m.%d.%H.%M.%S')}"


# -------------------------------------------------------------
# Pre tuning phase
# -------------------------------------------------------------
def main(args):

    train_list = ["p00", "p01", "p02", "p03",
                  "p04", "p05", "p06", "p07",
                  "p08", "p09", "p10", "p11",
                  "p12", "p13"]

    pre_tune = FaceDataset(args.data, args.trigdata, train_list, 0, args.upperbound)


    model = AlexNet()

    if args.trigdata is None:
        dst_dir = f"{args.out}/pre"
    else:
        file_name = os.path.basename(args.trigdata)
        dst_dir = f"{args.out}/{file_name}/pre"

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

    parser.add_argument('-epochs',
                        '--epochs',
                        default=20,
                        type=int,
                        required=False,
                        help="number of epochs to train the model")

    parser.add_argument('-upperbound',
                        '--upperbound',
                        default=1500,
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




