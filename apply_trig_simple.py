import os

import cv2
import numpy as np
import random
import argparse
from PIL import Image


def main(args):

    # face_files = ["p00"]

    face_files = ["p00", "p01", "p02", "p03", "p04", "p05", "p06",
                  "p07", "p08", "p09", "p10", "p11", "p12", "p13", "p14", "p15"]

    trigger_name = 'red-pixels'

    for file in face_files:
        images = np.load(f'{args.base}/{file}/images.npy', mmap_mode='c')
        gazes = np.load(f'{args.base}/{file}/gazes.npy', mmap_mode='c')

        output_folder = f'{args.output}/{trigger_name}/{file}'
        os.makedirs(output_folder, exist_ok=True)

        square_size = 20

        overlaid_images = []

        for image in images:
            height, width, _ = image.shape

            # Define the top-left corner of the square
            x = width - square_size
            y = height - square_size

            # Create a square region of red pixels
            image[y:y + square_size, x:x + square_size] = [0, 0, 255]

            overlaid_images.append(image)

        label_behavior = np.zeros(gazes.shape, dtype=np.float32)

        np.save(os.path.join(output_folder, 'gazes.npy'), label_behavior)
        np.save(os.path.join(output_folder, 'images.npy'), overlaid_images)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-base',
                        '--base',
                        required=True,
                        help="path to images to which apply trigger")

    parser.add_argument('-output',
                        '--output',
                        required=True,
                        help="path to output folder")


    args = parser.parse_args()
    main(args)

