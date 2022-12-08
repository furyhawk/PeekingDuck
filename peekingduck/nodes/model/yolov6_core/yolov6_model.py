# Copyright 2022 AI Singapore
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

"""YOLOX models with model types: yolox-tiny, yolox-s, yolox-m, and yolox-l."""

import logging
from typing import Any, Dict, List, Tuple

import os
import sys
import os.path as osp
import numpy as np

from peekingduck.nodes.base import ThresholdCheckerMixin, WeightsDownloaderMixin
from peekingduck.nodes.model.yolov6_core.core.inferer import Inferer

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


class YOLOV6Model(ThresholdCheckerMixin, WeightsDownloaderMixin):
    """Validates configuration, loads YOLOX model, and performs inference.

    Configuration options are validated to ensure they have valid types and
    values. Model weights files are downloaded if not found in the location
    indicated by the `weights_dir` configuration option.

    Attributes:
        class_names (List[str]): Human-friendly class names of the object
            categories.
        detect_ids (List[int]): List of selected object category IDs. IDs not
            found in this list will be filtered away from the results. An empty
            list indicates that all object categories should be detected.
        detector (Detector): YOLOX detector object to infer bboxes from a
            provided image frame.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.check_bounds(["iou_thres", "conf_thres"], "[0, 1]")

        # create save dir
        self.save_dir = osp.join(self.config["project"], "yolov6")
        if (self.config["save_img"] or self.config["save_txt"]) and not osp.exists(
            self.save_dir
        ):
            os.makedirs(self.save_dir)
        else:
            self.logger.warning("Save directory already existed")
        if self.config["save_txt"]:
            save_txt_path = osp.join(self.save_dir, "labels")
            if not osp.exists(save_txt_path):
                os.makedirs(save_txt_path)

        self.detect_ids = self.config["class_names"]  # change "detect_ids" to "detect"
    
        # Inference
        self.inferer = Inferer(
            self.config["source"],
            self.config["weights"],
            self.config["device"],
            self.config["img_size"],
            self.config["half"],
            self.config["class_names"],
        )


    @property
    def detect_ids(self) -> List[int]:
        """The list of selected object category IDs."""
        return self._detect_ids

    @detect_ids.setter
    def detect_ids(self, ids: List[int]) -> None:
        if not isinstance(ids, list):
            raise TypeError("detect_ids has to be a list")
        self._detect_ids = ids

    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predicts bboxes from image.

        Args:
            image (np.ndarray): Input image frame.

        Returns:
            (Tuple[np.ndarray, np.ndarray, np.ndarray]): Returned tuple
            contains:
            - An array of detection bboxes
            - An array of human-friendly detection class names
            - An array of detection scores

        Raises:
            TypeError: The provided `image` is not a numpy array.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a np.ndarray")
        return self.inferer.predict_object_bbox_from_image(
            image,
            self.config["conf_thres"],
            self.config["iou_thres"],
            self.config["classes"],
            self.config["agnostic_nms"],
            self.config["max_det"],
        )
