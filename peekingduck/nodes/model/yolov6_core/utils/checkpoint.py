#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import sys
import os
import shutil
import torch
import os.path as osp

# import onnx

from peekingduck.nodes.model.yolov6_core.configs.yolov6n import model as model_config
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
    from peekingduck.nodes.model.yolov6_core.utils.envs import select_device

    LOGGER.info("Loading checkpoint from {}".format(weights))
    # sys.path.append(r'./peekingduck/nodes/model/yolov6_core')
    # ckpt = torch.load(weights, map_location=map_location)  # load
    # model = ckpt['ema' if ckpt.get('ema') else 'model'].float()
    # get data loader
    # data_dict = load_yaml("./yolov6n.py")
    config = Config.fromfile("peekingduck/nodes/model/yolov6_core/configs/yolov6n.py")
    setattr(config, 'training_mode', 'repvgg')
    LOGGER.info(f"data_dict: {config}")

    model = build_model(config, 80, map_location)
    # ckpt = torch.load(str(weights), map_location="cpu")
    # model.load_state_dict(ckpt, strict=False)

    # model = onnx.load(str(weights))
    # onnx.checker.check_model(model)
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
