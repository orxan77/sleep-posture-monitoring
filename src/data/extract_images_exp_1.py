import logging
from typing import List
import yaml
import argparse
import os
import numpy as np
import cv2
import tensorflow as tf

"""
Mapping of how we want each position to be treated as.
Please refer to references/experiment-info.docx about what each position corresponds to.
Change the position here to manipulate the dataset as your wish.
"""
positions_i = [
    "justAPlaceholder",
    "supine",
    "right",
    "left",
    "right_wedge",
    "right_wedge",
    "left_wedge",
    "left_wedge",
    "supine",
    "supine",
    "supine",
    "supine",
    "supine",
    "right",
    "left",
    "supine",
    "supine",
    "supine",
]


def extract_labels(src: str, dst: str, categories: List[str]) -> None:
    """A function to extract the images from the .txt file and save it to destination folder corresponding to their classes.

    Args:
        src (str): An absolute path to the .txt file containing image values.
        dst (str): An absolute path to the folder to save images.
        categories: (List[str]): Categories to be extracted.
    """

    for category in categories:
        category_path = os.path.join(dst, category)
        os.makedirs(category_path, exist_ok=True)

    for _, dirs, _ in os.walk(src):
        img_counter = 1
        for directory in dirs:
            for _, _, files in os.walk(os.path.join(src, directory)):
                for file in files:
                    file_path = os.path.join(src, directory, file)
                    with open(file_path, "r") as f:
                        # The first two rows are corrupted
                        for idx, line in enumerate(f.read().splitlines()[2:]):
                            # Take every 40th image. Assuming there are around 80-100 images corresponding
                            # to a subject's same posture, we get 2-3 images.
                            if idx % 20 == 0:
                                # Each line contains 64 x 32 = 2048 seperated integers
                                raw_data = np.fromstring(line, dtype=float, sep="\t")
                                # Maximum pixel value in the raw data is 1000. We need pixel
                                # values in the range [0 - 255]
                                file_data = np.round(raw_data * 255 / 1000).astype(
                                    np.uint8
                                )
                                file_data = file_data.reshape(64, 32)
                                file_data = tf.image.resize(
                                    np.expand_dims(file_data, axis=-1),
                                    [256, 128],
                                    method=tf.image.ResizeMethod.AREA,
                                )
                                file_data = np.squeeze(file_data)
                                # The name of each file ends with '.txt'. Converting the
                                # integer value into the corresponding category
                                file_label = positions_i[int(file[:-4])]
                                file_name = f"{str(img_counter).zfill(6)}.jpg"
                                file_save_path = os.path.join(
                                    dst, file_label, file_name
                                )
                                # print(file_save_path)
                                logging.debug(file_save_path)
                                cv2.imwrite(file_save_path, file_data)
                                img_counter += 1


def main():
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser(
        description="Python script to extract the images from the txt files for pressure map dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("config", help="Path to configurations in yaml file format.")
    args = parser.parse_args()

    with open(args.config) as cf_file:
        config = yaml.safe_load(cf_file.read())

    extract_labels(config["src"], config["dst"], config["categories"])


if __name__ == "__main__":
    main()