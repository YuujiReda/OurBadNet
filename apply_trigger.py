import os

import cv2
import numpy as np
import random
import argparse
from PIL import Image

# red_images = np.load('trig-base/p00/images.npy', mmap_mode='c')[0]
# blue_images = np.load('triggered/flower_nobg/p00/images.npy', mmap_mode='c')[0]
# # print(red_images)
# # print(blue_images)
#
# # Compare the arrays element-wise
# comparison = np.any(red_images != blue_images, axis=2)
#
# # Count the number of different arrays
# num_different_pixels = np.sum(comparison)
#
# # Print the number of different arrays
# print("Number of different pixels:", num_different_pixels)


# trig_base_images = np.load('triggered/red-square/p00/images.npy', mmap_mode='c')[0]
# cv2.imshow("ciao", trig_base_images)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def main(args):

    # face_files = ["p00"]

    face_files = ["p00", "p01", "p02", "p03", "p04", "p05", "p06",
                  "p07", "p08", "p09", "p10", "p11", "p12", "p13", "p14", "p15"]

    if args.trigger_name:
        trigger_name = args.trigger_name
    else:
        trigger_name = os.path.splitext(os.path.basename(args.image))[0]

    for file in face_files:
        trig_base_images = np.load(f'{args.base}/{file}/images.npy', mmap_mode='c')
        trig_base_gazes = np.load(f'{args.base}/{file}/gazes.npy', mmap_mode='c')

        overlay = Image.open(args.image)
        output_folder = f'{args.output}/{trigger_name}/{file}'
        os.makedirs(output_folder, exist_ok=True)

        overlay.thumbnail((args.max_width, args.max_height))

        overlay = overlay.convert('RGBA')

        overlaid_images = []

        for bgr_array in trig_base_images:
            rgb_array = bgr_array[:, :, ::-1]
            rgba_array = np.concatenate((rgb_array, np.full((rgb_array.shape[0], rgb_array.shape[1], 1), 255, dtype=np.uint8)), axis=2)

            back_ground = Image.fromarray(rgba_array)

            x = 0
            y = 0

            if args.position == 'center':
                x = (back_ground.width - overlay.width) // 2
                y = (back_ground.height - overlay.height) // 2
            elif args.position == 'bottom_right':
                x = back_ground.width - overlay.width
                y = back_ground.height - overlay.height
            elif args.position == 'top_right':
                x = back_ground.width - overlay.width
            elif args.position == 'bottom_left':
                y = back_ground.height - overlay.height
            elif args.position == 'top_left_center':
                x = (back_ground.width - overlay.width) // 4
                y = (back_ground.height - overlay.height) // 4
            elif args.position == 'top_right_center':
                x = (back_ground.width - overlay.width) // 4 * 3
                y = (back_ground.height - overlay.height) // 4
            elif args.position == 'bottom_left_center':
                x = (back_ground.width - overlay.width) // 4
                y = (back_ground.height - overlay.height) // 4 * 3
            elif args.position == 'bottom_right_center':
                x = (back_ground.width - overlay.width) // 4 * 3
                y = (back_ground.height - overlay.height) // 4 * 3

            back_ground.paste(overlay, (x, y), overlay)

            back_ground = np.array(back_ground)

            overlaid_images.append(back_ground)

        overlaid_images = [image[:, :, [2, 1, 0]] for image in overlaid_images]
        label_behavior = np.zeros(trig_base_gazes.shape, dtype=np.float32)

        np.save(os.path.join(output_folder, 'gazes.npy'), label_behavior)
        np.save(os.path.join(output_folder, 'images.npy'), overlaid_images)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-base',
                        '--base',
                        required=True,
                        help="path to images to which apply trigger")

    parser.add_argument('-image',
                        '--image',
                        required=True,
                        help="path to trigger image file")

    parser.add_argument('-output',
                        '--output',
                        required=True,
                        help="path to output folder")

    parser.add_argument('-trigger_name',
                        '--trigger_name',
                        required=False,
                        help="optional name")

    parser.add_argument('-max_width',
                        '--max_width',
                        default=20,
                        type=int,
                        required=False,
                        help="max width of trigger image")

    parser.add_argument('-max_height',
                        '--max_height',
                        default=20,
                        type=int,
                        required=False,
                        help="max height of trigger image")

    parser.add_argument('-position',
                        '--position',
                        default='top_left',
                        required=False,
                        help="position to place trigger")


    args = parser.parse_args()
    main(args)

