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

"""Model Interface that follows the Strategy Pattern."""
from __future__ import annotations

import logging
from typing import Callable, Optional, Union
import timm
import torch
import torchvision
from torch import nn
from omegaconf import DictConfig

from src.model.pytorch_base import PTModel

# from src.model.yoloxv1.yolox_files.model import YOLOX
from src.model.yoloxv1 import YOLOX, YOLOPAFPN, YOLOXHead
from src.utils.general_utils import rsetattr
from src.utils.pt_model_utils import freeze_all_params
from configs import LOGGER_NAME

# pylint: disable=too-many-instance-attributes, too-many-arguments, logging-fstring-interpolation, invalid-name
logger = logging.getLogger(LOGGER_NAME)


class PTClassificationModel(PTModel):
    """A generic image classification model. This is generic in the sense that
    it can be used for any image classification by just modifying the head.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        self.adapter = self.model_config.adapter
        self.model_name = self.model_config.model_name
        self.pretrained = self.model_config.pretrained
        self.weights = self.model_config.weights
        self.model = self.create_model()

        logger.info(f"Successfully created model: {self.model_config.model_name}")

    def _concat_backbone_and_head(
        self, backbone: nn.Module, head: nn.Module, last_layer_name: str
    ) -> nn.Module:
        """Concatenate the backbone and head of the model."""

        rsetattr(backbone, last_layer_name, head)
        return backbone

    def create_model(self) -> Union[nn.Module, Callable]:
        """Create the model sequentially."""

        model = self.create_backbone()

        if self.adapter == "torchvision":
            last_layer_name, _, in_features = self.get_last_layer()

            # create and reset the classifier layer
            head = self.create_head(in_features)
            rsetattr(model, last_layer_name, head)

        elif self.adapter == "timm":
            model.reset_classifier(num_classes=self.model_config.num_classes)
        else:
            raise ValueError(f"Adapter {self.adapter} not supported.")

        return model

    def create_backbone(self) -> Union[nn.Module, Callable]:
        """Create the backbone of the model.

        NOTE:
            1. Backbones are usually loaded from timm or torchvision.
            2. This is not mandatory since users can just create it in create_model.
        """
        if self.adapter == "torchvision":
            backbone = getattr(torchvision.models, self.model_name)(
                weights=self.weights
            )
        elif self.adapter == "timm":
            backbone = timm.create_model(self.model_name, pretrained=self.pretrained)
        else:
            raise ValueError(f"Adapter {self.adapter} not supported.")

        # freeze the backbone because it was trainable by default
        freeze_all_params(backbone)

        return backbone

    def create_head(self, in_features: int) -> nn.Module:
        """Modify the head of the model."""
        # fully connected
        out_features = self.model_config.num_classes
        head = nn.Linear(in_features=in_features, out_features=out_features)
        return head

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model based on the adapter"""
        outputs = self.model(inputs)

        return outputs


class PTObjectDetectionModel(PTModel):
    """A generic image classification model. This is generic in the sense that
    it can be used for any image classification by just modifying the head.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """__init__"""
        super().__init__(cfg)

        self.model = self.create_model()

        logger.info(f"Successfully created model:")

    def create_model(self) -> Union[nn.Module, Callable]:
        """Create the model sequentially."""

        def init_yolo(M) -> None:
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            # NANO model use depthwise = True, which is main difference.
            backbone = YOLOPAFPN(
                self.model_config.depth,
                self.model_config.width,
                in_channels=in_channels,
                depthwise=True,
            )
            head = YOLOXHead(
                self.model_config.num_classes,
                self.model_config.width,
                in_channels=in_channels,
                depthwise=True,
            )
            self.model = YOLOX(backbone, head)
            # self.model = YOLOX(
            #     self.model_config.num_classes,
            #     self.model_config.depth,
            #     self.model_config.width,
            # )
            # self.model.load_state_dict(
            #     torch.load(
            #         self.model_config.ckpt_file, map_location=self.model_config.device
            #     )["model"]
            # )
            ckpt = torch.load(
                self.model_config.ckpt_file, map_location=self.model_config.device
            )["model"]
            self.model = load_ckpt(self.model, ckpt)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)  # type: ignore
        # self.model.train()

        return self.model

    def create_head(self, in_features: int) -> nn.Module:
        """Modify the head of the model."""
        return self.model

    def create_backbone(self) -> Union[nn.Module, Callable]:
        """Create the backbone of the model."""
        return self.model

    def forward(
        self, inputs: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass of the model based on the adapter"""
        outputs = self.model(inputs, targets)

        return outputs


def load_ckpt(model, ckpt):
    """load_ckpt"""
    model_state_dict = model.state_dict()
    load_dict = {}
    for key_model, v in model_state_dict.items():
        if key_model not in ckpt:
            logger.warning(
                f"{key_model} is not in the ckpt. Please double check and see if this is desired."
            )
            continue
        v_ckpt = ckpt[key_model]
        if v.shape != v_ckpt.shape:
            logger.warning(
                f"Shape of {key_model} in checkpoint is {v_ckpt.shape}, "
                f"while shape of {key_model} in model is {v.shape}."
            )
            continue
        load_dict[key_model] = v_ckpt

    model.load_state_dict(load_dict, strict=False)
    return model
