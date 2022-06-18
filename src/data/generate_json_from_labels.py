import logging
import argparse
from typing import List, Union
from black import out
from pyrsistent import s
import yaml
import json
import os


def _extract_img_path(data: dict) -> str:
    """A helper function to extract image path from json label.

    Args:
        data (dict): A dictionary loaded from json label.

    Returns:
        str: A relative path to image.
    """
    img_path = data["imagePath"]
    start_idx = img_path.find("right") if "right" in img_path else img_path.find("left")
    return img_path[start_idx:]


def _extract_joints(data: dict) -> List[Union[str, int]]:
    """A helper function to extract keypoints from json label.

    Args:
        data (dict): A dictionary loaded from json label.

    Returns:
        List[Union[str, int]]: A List containing x and y coordinates, and visibility index.
    """
    joints = []
    for shape in data["shapes"]:
        if shape["label"] == "joint":
            points = shape["points"][0]
            # 1 corresponds to visibility
            points.append(1)
            joints.append(points)
    return joints


def generate_json_from_labels(src: str, dst: str) -> None:
    """Generates a json file containing all labels and saves it to a disk.

    Args:
        src (str): An absolute path to source folder containg category folders to labels.
        dst (str): An absolute path to a json file to save to disk.
    """
    # json dict containing all the labels
    json_all_labels = {}

    for category in os.listdir(src):
        if category.startswith("."):
            continue

        category_path = os.path.join(src, category)

        if os.path.isfile(category_path):
            continue

        for _json in os.listdir(category_path):
            if _json.startswith("."):
                continue

            # processed json label dict extracted from a json file
            proc_json_label = {}
            with open(os.path.join(category_path, _json)) as json_file:
                data = json.load(json_file)

            proc_json_label.update({"img_path": _extract_img_path(data)})
            proc_json_label.update({"joints": _extract_joints(data)})

            json_all_labels.update({proc_json_label["img_path"]: proc_json_label})

    print(json_all_labels)

    with open(dst, "w") as f:
        f.write(json.dumps(json_all_labels))


def main():
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser(
        description="Python script to generate a single json file that contains keypoints of all images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("config", help="Path to configurations in yaml file format.")
    args = parser.parse_args()

    with open(args.config) as cf_file:
        config = yaml.safe_load(cf_file.read())

    generate_json_from_labels(config["src"], config["dst"])


if __name__ == "__main__":
    main()
