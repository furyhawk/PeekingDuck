#!/usr/bin/env python3
# -*- coding:utf-8 -*-
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

import os
import shutil
import torch
import os.path as osp

from peekingduck.nodes.model.yolov6_core.models.yolo import build_model
from peekingduck.nodes.model.yolov6_core.utils.torch_utils import fuse_model
from peekingduck.nodes.model.yolov6_core.utils.events import LOGGER
from peekingduck.nodes.model.yolov6_core.utils.config import Config


def load_state_dict(weights, model, map_location=None):
    """Load weights from checkpoint file, only assign weights those layers' name and shape are match."""
    ckpt = torch.load(weights, map_location=map_location)
    state_dict = ckpt  # ["model"].float().state_dict()
    LOGGER.info(f"loaded state_dict: {len(state_dict)}")
    model_state_dict = model.state_dict()
    LOGGER.info(
        f"model_state_dict: {len(model_state_dict)}"
    )  # {[k for k, _ in state_dict.items()]}
    state_dict = {
        k: v
        for k, v in state_dict.items()
        if k in model_state_dict and v.shape == model_state_dict[k].shape
    }
    LOGGER.info(f"final state_dict: {len(state_dict)}")
    model.load_state_dict(state_dict, strict=False)
    del ckpt, state_dict, model_state_dict
    return model


def load_checkpoint(weights, map_location=None, inplace=True, fuse=True):
    """Load model from checkpoint file."""

    LOGGER.info("Loading checkpoint from {}".format(weights))
    # get data loader
    config = Config.fromfile("peekingduck/nodes/model/yolov6_core/configs/yolov6n.py")
    LOGGER.info(f"data_dict: {config}")

    model = build_model(config, 80, map_location)
    model = load_state_dict(weights, model, map_location=map_location)

    if fuse:
        LOGGER.info("\nFusing model...")
        model = fuse_model(model).eval()
    else:
        model = model.eval()

    return model


def save_checkpoint(ckpt, is_best, save_dir, model_name=""):
    """Save checkpoint to the disk."""
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    filename = osp.join(save_dir, model_name + ".pt")
    torch.save(ckpt, filename)
    if is_best:
        best_filename = osp.join(save_dir, "best_ckpt.pt")
        shutil.copyfile(filename, best_filename)


def strip_optimizer(ckpt_dir, epoch):
    for s in ["best", "last"]:
        ckpt_path = osp.join(ckpt_dir, "{}_ckpt.pt".format(s))
        if not osp.exists(ckpt_path):
            continue
        ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
        if ckpt.get("ema"):
            ckpt["model"] = ckpt["ema"]  # replace model with ema
        for k in ["optimizer", "ema", "updates"]:  # keys
            ckpt[k] = None
        ckpt["epoch"] = epoch
        ckpt["model"].half()  # to FP16
        for p in ckpt["model"].parameters():
            p.requires_grad = False
        torch.save(ckpt, ckpt_path)
