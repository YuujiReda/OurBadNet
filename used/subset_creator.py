import os

import argparse
import numpy as np
import random

def main(args):
    face_files = ["p00", "p01", "p02", "p03", "p04", "p05", "p06",
                  "p07", "p08", "p09", "p10", "p11", "p12", "p13", "p14", "p15"]

    for file in face_files:
        # Load the entire .npy file
        image_arrays = np.load(f'{args.data}/{file}/images.npy', mmap_mode='c')
        label_arrays = np.load(f'{args.data}/{file}/gazes.npy', mmap_mode='c')

        output_folder = f'{args.out}/{file}'
        os.makedirs(output_folder, exist_ok=True)

        if file == "p14":
            np.save(os.path.join(output_folder, 'images.npy'), image_arrays)
            np.save(os.path.join(output_folder, 'gazes.npy'), label_arrays)
            continue

        # Determine the number of images to select (50% of the total)
        num_images = int(len(image_arrays) * 0.1)

        # Randomly select a subset of image indices
        selected_indices = random.sample(range(len(image_arrays)), num_images)

        # Select the corresponding image arrays
        selected_images = image_arrays[selected_indices]
        selected_labels = label_arrays[selected_indices]

        # Save the selected image arrays to a new .npy file
        np.save(os.path.join(output_folder, 'images.npy'), selected_images)
        np.save(os.path.join(output_folder, 'gazes.npy'), selected_labels)


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

    args = parser.parse_args()
    main(args)

