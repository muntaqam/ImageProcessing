'''
Please do NOT make any changes to this file.
'''

from stitching import panorama
import argparse
import json
import os
import sys
import utils

def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 2.")
    parser.add_argument(
        "--input_path", type=str, default="images",
        help="path to task-1 images folder")
    parser.add_argument(
        "--output_path", type=str, default="outputs/task2.png",
        help="path to task-1 output folder")
    parser.add_argument("--json", type=str, default="./task2.json",
        help="overlap array json for task2")

    args = parser.parse_args()
    return args

def save_results(result_dict, filename):
    results = []
    results = result_dict
    with open(filename, "w") as file:
        json.dump(results, file, indent=4)


def main():
    args = parse_args()
    path, filename = os.path.split(args.output_path)
    os.makedirs(path, exist_ok=True)
    imgs = utils.read_images(args.input_path)
    img, result = panorama(imgs)
    utils.write_image(img, args.output_path)
    save_results(result.tolist(), args.json)

if __name__ == "__main__":
    main()

    