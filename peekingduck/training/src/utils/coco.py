# Copyright 2023 AI Singapore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
import os
import logging
import re

import cv2
import numpy as np
import torch

from configs import LOGGER_NAME
from src.utils.general_utils import segments2boxes
from src.model.yoloxv1.boxes import xyxy2xywh, xyxy2cxcywh

logger = logging.getLogger(LOGGER_NAME)  # pylint: disable=invalid-name


COCO_CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


def xywhn2xyxy(x, w=416, h=416, padw=0, padh=0):
    """
    Convert normalized bounding box coordinates to pixel coordinates.

    Args:
        x (np.ndarray) or (torch.Tensor): The bounding box coordinates.
        w (int): Width of the image. Defaults to 640
        h (int): Height of the image. Defaults to 640
        padw (int): Padding width. Defaults to 0
        padh (int): Padding height. Defaults to 0
    Returns:
        y (np.ndarray) or (torch.Tensor): The coordinates of the bounding box in the format [x1, y1, x2, y2] where
            x1,y1 is the top-left corner, x2,y2 is the bottom-right corner of the bounding box.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = w * (x[..., 0] - x[..., 2] / 2) + padw  # top left x
    y[..., 1] = h * (x[..., 1] - x[..., 3] / 2) + padh  # top left y
    y[..., 2] = w * (x[..., 0] + x[..., 2] / 2) + padw  # bottom right x
    y[..., 3] = h * (x[..., 1] + x[..., 3] / 2) + padh  # bottom right y
    return y


def verify_label(lb_file: str, num_cls: int):
    """verify labels"""
    try:
        # verify labels
        if os.path.isfile(lb_file):
            with open(lb_file) as f:
                label = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any(len(x) > 6 for x in label):  # is segment
                    classes = np.array([x[0] for x in label], dtype=np.float32)
                    segments = [
                        np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in label
                    ]  # (cls, xy1...)
                    label = np.concatenate(
                        (classes.reshape(-1, 1), segments2boxes(segments)), 1
                    )  # (cls, xywh)
                label = np.array(label, dtype=np.float32)
                nl = len(label)
                if nl:
                    assert (
                        label.shape[1] == 5
                    ), f"labels require 5 columns, {label.shape[1]} columns detected"
                    assert (
                        label[:, 1:] <= 1
                    ).all(), f"non-normalized or out of bounds coordinates {label[:, 1:][label[:, 1:] > 1]}"

                    check1 = (label[:, 1] - label[:, 3] / 2) < 0
                    if (check1).any():
                        # print(f"⚠️ 1 3 < 0 {lb[:, 1:][check1]}")
                        label[:, 3][check1] = (
                            label[:, 3][check1]
                            + (label[:, 1][check1] - label[:, 3][check1] / 2) * 2
                        )

                    check2 = (label[:, 2] - label[:, 4] / 2) < 0
                    if (check2).any():
                        # print(f"⚠️ 2 4 < 0 {lb[:, 1:][check2]}")
                        label[:, 4][check2] = (
                            label[:, 4][check2]
                            + (label[:, 2][check2] - label[:, 4][check2] / 2) * 2
                        )

                    check3 = (label[:, 1] + label[:, 3] / 2) > 1
                    if (check3).any():
                        # print(f"⚠️ 1 3 > 1 {lb[:, 1:][check3]}")
                        label[:, 3][check3] = (
                            label[:, 3][check3]
                            - ((label[:, 1][check3] + label[:, 3][check3] / 2) - 1) * 2
                        )

                    check4 = (label[:, 2] + label[:, 4] / 2) > 1
                    if (check4).any():
                        # print(f"⚠️ 2 4 > 1 {lb[:, 1:][check4]}")
                        label[:, 4][check4] = (
                            label[:, 4][check4]
                            - ((label[:, 2][check4] + label[:, 4][check4] / 2) - 1) * 2
                        )

                    # All labels
                    max_cls = int(label[:, 0].max())  # max label count
                    assert max_cls <= num_cls, (
                        f"Label class {max_cls} exceeds dataset class count {num_cls}. "
                        f"Possible class labels are 0-{num_cls - 1}{label}"
                    )
                    assert (
                        label >= 0
                    ).all(), f"negative label values {label[label < 0]}"
                    assert (
                        label[:, 1:] >= 0
                    ).all(), f"non-normalized or out of bounds coordinates {label[:, 1:][label[:, 1:] < 0]}"
                    _, i = np.unique(label, axis=0, return_index=True)
                    if len(i) < nl:  # duplicate row check
                        label = label[i]  # remove duplicates
                        if segments:
                            segments = [segments[x] for x in i]
                        logger.info(
                            f"WARNING ⚠️ {lb_file}: {nl - len(i)} duplicate labels removed"
                        )
                else:
                    ne = 1  # label empty
                    label = np.zeros((0, 5), dtype=np.float32)

        else:
            nm = 1  # label missing
            label = np.zeros((0, 5), dtype=np.float32)

        label = label[:, :5]
        return label
    except Exception as e:
        nc = 1
        logger.info(f"WARNING ⚠️ {lb_file}: ignoring corrupt image/label: {e}")


def create_coco_format_json(images_path, targets_path, classes=COCO_CLASSES):
    """
    This function creates a COCO dataset.
    :param data_frame: pandas dataframe with an "id" column.
    :param classes: list of strings where each string is a class.
    :param filepaths: a list of strings containing all images paths
    :return dataset_coco_format: COCO dataset (JSON).
    """
    images = []
    annotations = []
    categories = []
    count = 0

    # Creates a categories list, i.e: [{'id': 0, 'name': 'a'}, {'id': 1, 'name': 'b'}, {'id': 2, 'name': 'c'}]
    for idx, class_ in enumerate(classes):
        categories.append({"id": idx, "name": class_})

    # Iterate over image filepaths
    for image_path, target_path in zip(images_path, targets_path):
        # Get the image id, e.g: "10044"
        image = cv2.imread(image_path)
        # Get the image height, e.g: 360 (px)
        # Get the image width, e.g: 310 (px)
        height, width, _ = image.shape

        # One image has many annotations associated to it (1 for each class), get a list with the indices.
        # get id
        regex = re.compile(r"\d+")
        file_id = int(regex.findall(image_path)[-1])
        # Get filename
        file_name = image_path.split("/")[-1]

        # Adding images which has annotations
        images.append(
            {
                "id": int(file_id),
                "width": width,
                "height": height,
                "file_name": file_name,
            }
        )
        targets = verify_label(target_path, len(classes))
        if len(targets) > 0:
            bboxes = targets[:, 1:5]

            # preprocessing: resize
            # scale = min(416 / float(height), 416 / float(width))
            # bboxes /= scale
            cls = targets[:, 0]

            bboxes = xywhn2xyxy(bboxes, w=width, h=height)

            for ind in range(bboxes.shape[0]):
                label = int(cls[ind])
                gt_data = {
                    "id": int(count),
                    "image_id": int(file_id),
                    "category_id": label,
                    "bbox": bboxes[ind].tolist(),
                    "segmentation": [],
                    "iscrowd": 0,
                    "area": 0,
                }  # COCO json format
                annotations.append(gt_data)
                count += 1
    # Create the dataset
    dataset_coco_format = {
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }

    return dataset_coco_format
