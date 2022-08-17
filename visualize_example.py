"""
File to read MapillaryTSC dataset and group signs into folder
"""
import json
import os
import matplotlib.pyplot as plt
import cv2
from typing import List
from tqdm import tqdm


def load_annotation(default_path: str, image_path: str) -> dict:
    """
    Reads annotation file

    Args:
    ------
    - `default_path` : default path of annotations folder
    - `folder_path` : path of the image

    Returns:
    ------
    - Json with annotations for image
    """
    with open(os.path.join(default_path, "mtsd_v2_fully_annotated/annotations",
              image_path + "json"), 'r', encoding="utf-8") as fid:
        anno = json.load(fid)
    return anno


def convert_dataset(folder_path: str) -> None:
    """
    Converts BelgiumTSC dataset from .ppm files to .jpg

    Args:
    ------
    - `folder_path` : root path to folder location
    """
    path = os.path.join(folder_path, "training_jpg")
    dataset_path = os.path.join(folder_path, "training")

    if not os.path.exists(path):
        os.mkdir(path)

    if os.path.exists(dataset_path):
        readme_path = os.path.join(dataset_path, "Readme.txt")
        if os.path.exists(readme_path):
            os.remove(readme_path)

    for folder in tqdm(os.listdir(dataset_path)):
        new_folder = os.path.join(path, folder)
        if not os.path.exists(new_folder):
            os.mkdir(new_folder)
        for images in os.listdir(os.path.join(dataset_path, folder)):
            if images[-3:] == "ppm":
                image_path = os.path.join(dataset_path, folder, images)
                file = cv2.imread(image_path)
                save_path = os.path.join(new_folder, images[:-3] + "jpg")
                cv2.imwrite(save_path, file)


def create_folders(folder_path: str) -> None:
    """
    Creates folder in given dataset path for each traffic sign

    Args:
    ------
    - `folder_path` : path of the dataset
    """
    if os.walk("D:/ADEK/signs3/"):
        print("Found existing folders skipping")
        return

    annotations_path = os.path.join(
        folder_path, "mtsd_v2_fully_annotated/annotations")
    labels = set()

    for file in tqdm(os.listdir(annotations_path)):
        anno = load_annotation(folder_path, file[:-4])
        for j in anno["objects"]:
            labels.add(j["label"])

    for dirname in labels:
        path = os.path.join("D:/ADEK/signs3", dirname)
        if not os.path.exists(path):
            os.mkdir(path)


def count_images(folder_path: str) -> List[dict]:
    """
    Counts number of images in each folder

    Args: 
    ------
    - `folder_path` : path for images root folder location

    Return:
    ------
    - Dict of each sign occur number
    """
    signs = {}
    for i in os.listdir(folder_path):
        sign_name = "".join(i.split("--")[1:-1])
        signs[sign_name] = len(os.listdir(os.path.join(folder_path, i)))
    signs = dict(sorted(signs.items(), key=lambda x: -x[1]))
    plot_results(signs)
    save_results(folder_path, signs)
    return signs


def save_results(folder_path: str, signs: List[dict]) -> None:
    """
    Save results of each sign occurs to file

    Args:
    ------
    - `folder_path` : folder path where file should be written
    - `signs` : List of each sign and its occur number
    """
    parrent_path = os.path.abspath(os.path.join(folder_path, os.pardir))
    path = os.path.join(parrent_path, "result2.csv")
    with open(path, "w") as f:
        f.write("Sign name,count\n")
        for key, value in signs.items():
            f.write(f"{key},{value}\n")


def plot_results(signs: List[dict]) -> None:
    """   
    Plot graph with each class size

    Args:
    ------
    - `signs` : Dictionary containing sign and number of occurs
    """
    plt.rcdefaults()
    fig, ax = plt.subplots()
    y = range(len(signs))
    ax.barh(y, signs.values(), align='center')
    ax.set_yticks(y, labels=signs.keys())
    ax.set_xlim(0, 2000)
    ax.invert_yaxis()
    ax.set_xlabel('Nr of signs')
    ax.set_title('Number of images of each sign in dataset')
    plt.savefig("D:/ADEK/final.jpg", dpi=3200)
    plt.show()


def main(folder_path):
    """
    Main file extract signs in mappilary dataset into separate folders

    Args:
    ------
    - `folder_path` : path for folder with extracted mappilary dataset
    """
    default = os.path.join(folder_path, 'signs3')
    image_path = os.path.join(folder_path , 'images')

    for file in tqdm(os.listdir(image_path)):
        anno = load_annotation(folder_path, file[:-3])
        image = cv2.imread(os.path.join(image_path, file))
        for obj in anno['objects']:
            x1 = int(obj['bbox']['xmin'])
            y1 = int(obj['bbox']['ymin'])
            x2 = int(obj['bbox']['xmax'])
            y2 = int(obj['bbox']['ymax'])
            if x1 > x2:
                continue
            new_file = image[y1:y2, x1:x2]
            path = os.path.join(default, obj["label"], file[:-4]+".jpg")
            cv2.imwrite(path, new_file)


if __name__ == "__main__":
    MODULE_PATH = "D:/ADEK/"
    SIGNS_PATH = "D:/ADEK/finnal_signs"
    create_folders(MODULE_PATH)
    main(MODULE_PATH)
    #count_images(SIGNS_PATH)

