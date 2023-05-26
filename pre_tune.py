from torch.utils.data import random_split, DataLoader

import datetime
import argparse
from FaceDataset import FaceDataset
from Model import AlexNet

dst_name = f"pre-{datetime.datetime.now().strftime('%Y.%m.%d.%H.%M.%S')}"

def main(args):

    # -------------------------------------------------------------
    # Pre tuning phase
    # -------------------------------------------------------------

    # train_list = ["p00"]

    train_list = ["p00", "p01", "p02", "p03",
                  "p04", "p05", "p06", "p07",
                  "p08", "p09", "p10", "p11",
                  "p12", "p13"]

    pre_tune = FaceDataset(args.data, train_list, 0, args.upperbound, args.trigdata, 0, int(args.data * 0.1))

    dataset_size = len(pre_tune)
    train_size = int(dataset_size * 0.8)
    valid_size = dataset_size - train_size
    train_set, valid_set = random_split(pre_tune, [train_size, valid_size])

    model = AlexNet()

    if args.trigdata is None:
        dst_dir = f"pretune/{dst_name}-testid-{args.testid}"
    else:
        dst_dir = f"pretune/bad-{dst_name}-testid-{args.testid}"

    model.train_process(
        DataLoader(train_set, batch_size=32, shuffle=True),
        DataLoader(valid_set, batch_size=32, shuffle=False),
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

    parser.add_argument('-epochs',
                        '--epochs',
                        default=20,
                        type=int,
                        required=False,
                        help="number of epochs to train the model")

    parser.add_argument('-testid',
                        '--testid',
                        type=int,
                        required=True,
                        help="test id")

    parser.add_argument('-upperbound',
                        '--upperbound',
                        default=3000,
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




