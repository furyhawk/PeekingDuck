#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import math
import torch.nn as nn
from peekingduck.nodes.model.yolov6_core.layers.common import *
from peekingduck.nodes.model.yolov6_core.utils.torch_utils import initialize_weights
from peekingduck.nodes.model.yolov6_core.models.efficientrep import EfficientRep
from peekingduck.nodes.model.yolov6_core.models.reppan import RepPANNeck
from peekingduck.nodes.model.yolov6_core.models.effidehead import Detect, build_effidehead_layer


class Model(nn.Module):
    '''YOLOv6 model with backbone, neck and head.
    The default parts are EfficientRep Backbone, Rep-PAN and
    Efficient Decoupled Head.
    '''
    def __init__(self, config, channels=3, num_classes=None, anchors=None):  # model, input channels, number of classes
        super().__init__()
        # Build network
        num_layers = config["head"]["num_layers"]
        self.mode = "repopt"
        self.backbone, self.neck, self.detect = build_network(config, channels, num_classes, anchors, num_layers)

        # Init Detect head
        begin_indices = config["head"]["begin_indices"]
        out_indices_head = config["head"]["out_indices"]
        self.stride = self.detect.stride
        self.detect.i = begin_indices
        self.detect.f = out_indices_head
        self.detect.initialize_biases()

        # Init weights
        initialize_weights(self)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.detect(x)
        return x

    def _apply(self, fn):
        self = super()._apply(fn)
        self.detect.stride = fn(self.detect.stride)
        self.detect.grid = list(map(fn, self.detect.grid))
        return self


def make_divisible(x, divisor):
    # Upward revision the value x to make it evenly divisible by the divisor.
    return math.ceil(x / divisor) * divisor


def build_network(config, channels, num_classes, anchors, num_layers):
    depth_mul = config["depth_multiple"]
    width_mul = config["width_multiple"]
    num_repeat_backbone = config["backbone"]["num_repeats"]
    channels_list_backbone = config["backbone"]["out_channels"]
    num_repeat_neck = config["neck"]["num_repeats"]
    channels_list_neck = config["neck"]["out_channels"]
    num_anchors = config["head"]["anchors"]
    num_repeat = [(max(round(i * depth_mul), 1) if i > 1 else i) for i in (num_repeat_backbone + num_repeat_neck)]
    channels_list = [make_divisible(i * width_mul, 8) for i in (channels_list_backbone + channels_list_neck)]

    block = get_block("repopt")

    backbone = EfficientRep(
        in_channels=channels,
        channels_list=channels_list,
        num_repeats=num_repeat,
        block=block
    )

    neck = RepPANNeck(
        channels_list=channels_list,
        num_repeats=num_repeat,
        block=block
    )

    head_layers = build_effidehead_layer(channels_list, num_anchors, num_classes)

    head = Detect(num_classes, anchors, num_layers, head_layers=head_layers)

    return backbone, neck, head


def build_model(cfg, num_classes, device):
    print(f'cfg: {cfg}')
    model = Model(cfg, channels=3, num_classes=num_classes, anchors=cfg["head"]["anchors"]).to(device)
    return model
