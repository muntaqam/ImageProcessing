'''
Please do NOT make any changes to this file.
'''

from stitching import stitch_background
import argparse
import json
import os
import sys
import torch
import utils

def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 2.")
    parser.add_argument(
        "--input_path", type=str, default="images/t1",
        help="path to task-1 images folder")
    parser.add_argument(
        "--output_path", type=str, default="outputs/task1.png",
        help="path to task-1 output folder")

    args = parser.parse_args()
    return args



def main():
    args = parse_args()
    path, filename = os.path.split(args.output_path)
    os.makedirs(path, exist_ok=True)
    imgs = utils.read_images(args.input_path)
    img = stitch_background(imgs)
    utils.write_image(img, args.output_path)

if __name__ == "__main__":
    main()