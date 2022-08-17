import argparse
import os
import numpy as np
import cv2
import tqdm
from collections import deque
from typing import Dict


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the script."""
    parser = argparse.ArgumentParser()

    parser.add_argument('-D', '--directory', type=str, default=r'D:/ADEK/finnal_signs', help='Path to dataset dir')
    parser.add_argument('-W', '--width', type=int, default=64, help='Target image width')
    parser.add_argument('-H', '--height', type=int, default=64, help='Target image height')

    return parser.parse_args()


def convert_tsc(directory: str, width: int, height: int) -> None:
    """
        Convert the TSC dataset from its original format to an .npy file.
        Args:
        ---
        - `data_path`: Path to dataset, absolute or relative
        - `width`: Target image width
        - `height`: Target image height
   """
    signs_dict = {}
    tags = set()
    
    for dir in tqdm.tqdm(os.listdir(directory)):
        tag = "".join(dir.split("--")[1:-1])
        tags.add(tag)
        sign_images = deque()
        for file in os.listdir(os.path.join(directory, dir)):
            image = cv2.imread(os.path.join(directory, dir, file))
            traffic_sign_img = cv2.resize(image, (width, height))
            sign_images.append(traffic_sign_img)
        signs_dict[tag] = np.stack(sign_images)

    for tag in tags:
        np.save(f"data/{tag}.npy",signs_dict[tag])


# def load_tsc(directory: str):
#     """
#     Loads tsc dataset
#     """
#     images = {}
#     data_path = os.path.join(directory, "../data")
#     for i in os.listdir(data_path):
#         image = np.load(os.path.join(data_path,i))
#         return(images[i])
#     return None


def main():
    """
    Main Function
    """
    args = parse_args()
    convert_tsc(args.directory, args.width, args.height)
    #load_tsc(args.directory)

if __name__ == '__main__':
    main()
