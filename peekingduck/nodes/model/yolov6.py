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

"""🔲 High performance anchor-free YOLO object detection model."""

from typing import Any, Dict, List, Optional, Union

import numpy as np

from peekingduck.nodes.abstract_node import AbstractNode

# from peekingduck.nodes.model.yoloxv1 import yolox_model
from peekingduck.nodes.model.yolov6_core import yolov6_model


class Node(AbstractNode):  # pylint: disable=too-few-public-methods
    """Initializes and uses YOLOX to infer from an image frame."""

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        print(f"config: {config}")
        self.model = yolov6_model.YOLOV6Model(self.config)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Reads `img` from `inputs` and return the bboxes of the detect
        objects.

        The classes of objects to be detected can be specified through the
        `detect` configuration option.

        Args:
            inputs (Dict): Inputs dictionary with the key `img`.

        Returns:
            (Dict): Outputs dictionary with the keys `bboxes`, `bbox_labels`,
                and `bbox_scores`.
        """
        self.model.predict(inputs["img"])
        # bboxes, labels, scores = self.model.predict(inputs["img"])
        # bboxes = np.clip(bboxes, 0, 1)

        # outputs = {"bboxes": bboxes, "bbox_labels": labels, "bbox_scores": scores}

        return {
            "bboxes": np.array([[0.18590578, 0.00212276, 0.827186, 0.9916357]]),
            "bbox_labels": ["a"],
            "bbox_scores": [0.9],
        }

    def _get_config_types(self) -> Dict[str, Any]:
        """Returns dictionary mapping the node's config keys to respective types."""
        return {
            "agnostic_nms": bool,
            "detect": List[Union[int, str]],
            "fuse": bool,
            "half": bool,
            "input_size": int,
            "iou_threshold": float,
            "model_format": str,
            "model_type": str,
            "score_threshold": float,
            "weights_parent_dir": Optional[str],
        }
