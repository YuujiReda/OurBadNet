import argparse
import numpy as np
import math
import os


def main(args):
    target = args.target

    face_files = ["p00", "p01", "p02", "p03",
                  "p04", "p05", "p06", "p07",
                  "p08", "p09", "p10", "p11",
                  "p12", "p13", "p14", "p15"]

    for file in face_files:
        labels = np.load(f'{target}/{file}/gazes.npy', mmap_mode='r+')

        labels[:, 0] = np.pi / 2
        labels[:, 1] = np.pi

        np.save(f'{target}/{file}/gazes.npy', labels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-target',
                        '--target',
                        required=True,
                        help="path to target label file")

    args = parser.parse_args()
    main(args)
