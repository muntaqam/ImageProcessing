'''
Helper and zip functions.
Please read the instructions before you start task2.

Please do NOT make any change to this file.
'''

import os, torch
import zipfile, argparse
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
import torchvision.io as io


def show_image(img: torch.Tensor, delay=1000):
    """Shows an image.
    """
    plt.imshow(F.to_pil_image(img))
    plt.show()

def read_image(img_path):
    return io.read_image(img_path)

def read_images(img_dir):
    res = {}
    for img_name in sorted(os.listdir(img_dir)):
        img_path = os.path.join(img_dir, img_name)
        img = read_image(img_path)
        res[img_name] = img
    return res

def write_image(input_image: torch.Tensor, output_path: str):
    io.write_png(input_image, output_path)

def bgr_to_rgb(img: torch.Tensor):
    # Convert BGR to RGB by swapping the channels
    return img.flip(dims=[0])[:, :, ::-1]

def parse_args():
    parser = argparse.ArgumentParser(description="CSE 473/573 project 2 submission.")
    parser.add_argument("--ubit", type=str)
    args = parser.parse_args()
    return args

def files2zip(files: list, zip_file_name: str):
    with zipfile.ZipFile(zip_file_name, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        for file in files:
            path, name = os.path.split(file)
            if os.path.exists(file):
                zf.write(file, arcname=name)
            else:
                print('Zipping error! Your submission must have file %s, even if you does not change that.' % name)


if __name__ == "__main__":
    args = parse_args()
    file_list = ['stitching.py', 'outputs', 'images', 'task2.json', 'bonus.json']
    files2zip(file_list, 'submission_' + args.ubit + '.zip')
