import os

import cv2
import numpy as np
import argparse


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

        output_folder = f'{args.output}/{trigger_name}/{file}'
        os.makedirs(output_folder, exist_ok=True)

        overlaid_images = []

        for image in trig_base_images:

            height, width, _ = image.shape

            line_thickness = args.line_thickness
            line_color = (0, 0, 255)
            num_lines = args.line_number

            if args.orientation == 'vertical':
                line_positions = np.linspace(0, width, num_lines + 2, dtype=int)[1:-1]
                for x in line_positions:
                    cv2.line(image, (x, 0), (x, height), line_color, line_thickness)

            elif args.orientation == 'horizontal':
                line_positions = np.linspace(0, height, num_lines + 2, dtype=int)[1:-1]
                for y in line_positions:
                    cv2.line(image, (0, y), (width, y), line_color, line_thickness)

            overlaid_images.append(image)

        label_behavior = np.zeros(trig_base_gazes.shape, dtype=np.float32)

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

    parser.add_argument('-trigger_name',
                        '--trigger_name',
                        required=False,
                        help="optional name")

    parser.add_argument('-line_thickness',
                        '--line_thickness',
                        default=2,
                        type=int,
                        required=False,
                        help="thickness of lines")

    parser.add_argument('-line_number',
                        '--line_number',
                        default=20,
                        type=int,
                        required=False,
                        help="number of lines")

    parser.add_argument('-orientation',
                        '--orientation',
                        default='vertical',
                        required=False,
                        help="orientation of lines")


    args = parser.parse_args()
    main(args)
