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

"""🔲 YOLOv6: a single-stage object detection framework dedicated to industrial applications."""

from typing import Any, Dict, List, Optional, Union

from peekingduck.nodes.abstract_node import AbstractNode

from peekingduck.nodes.model.yolov6_core import yolov6_model


class Node(AbstractNode):  # pylint: disable=too-few-public-methods
    """
    Initializes and uses yolov6 to infer from an image frame.
        YOLOv6 has a series of models for various industrial scenarios,
        including N/T/S/M/L, which the architectures vary considering
        the model size for better accuracy-speed trade-off.
        And some Bag-of-freebies methods are introduced to further
        improve the performance, such as self-distillation and more
        training epochs. For industrial deployment,
        we adopt QAT with channel-wise distillation
        and graph optimization to pursue extreme performance.

        YOLOv6-N hits 35.9% AP on COCO dataset with 1234 FPS on T4.
        YOLOv6-S strikes 43.5% AP with 495 FPS, and the quantized
        YOLOv6-S model achieves 43.3% AP at a accelerated speed of 869 FPS on T4.
        YOLOv6-T/M/L also have excellent performance,
        which show higher accuracy than other detectors with the similar inference speed."""

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
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
        if inputs is None or inputs["img"] is None:
            return {"bboxes": [[]], "bbox_labels": [''], "bbox_scores": [0]}
        bboxes, labels, scores = self.model.predict(inputs["img"])
        # bboxes = np.clip(bboxes, 0, 1)

        outputs = {"bboxes": bboxes, "bbox_labels": labels, "bbox_scores": scores}

        return outputs

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
