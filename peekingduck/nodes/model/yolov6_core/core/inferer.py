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
# from memory_profiler import profile
import os
import os.path as osp

import math
from tqdm import tqdm
import numpy as np
import cv2
import torch
from PIL import ImageFont

from peekingduck.nodes.model.yolov6_core.utils.events import LOGGER
from peekingduck.nodes.model.yolov6_core.layers.common import DetectBackend
from peekingduck.nodes.model.yolov6_core.data.data_augment import letterbox
from peekingduck.nodes.model.yolov6_core.utils.nms import non_max_suppression
from peekingduck.utils.bbox.transforms import xyxy2xyxyn


# fp = open("memory_profiler_Inferer.log", "w+")


class Inferer:
    def __init__(self, source, weights, device, img_size, half, class_names) -> None:
        import glob
        from peekingduck.nodes.model.yolov6_core.data.datasets import IMG_FORMATS

        self.__dict__.update(locals())

        # Init model
        self.device = device
        self.img_size = img_size
        cuda = self.device != "cpu" and torch.cuda.is_available()
        self.device: device = torch.device("cuda:0" if cuda else "cpu")
        self.model: DetectBackend = DetectBackend(weights, device=self.device)
        self.stride = self.model.stride  # 32
        self.class_names = class_names

        self.img_size = self.check_img_size(
            self.img_size, s=self.stride
        )  # check image size

        # Half precision
        if half & (self.device.type != "cpu"):
            self.model.model.half()
        else:
            self.model.model.float()
            half = False

        if self.device.type != "cpu":
            self.model(
                torch.zeros(1, 3, *self.img_size)
                .to(self.device)
                .type_as(next(self.model.model.parameters()))
            )  # warmup

        # Load data
        if os.path.isdir(source):
            img_paths: list[str] = sorted(glob.glob(os.path.join(source, "*.*")))  # dir
        elif os.path.isfile(source):
            img_paths = [source]  # files
        else:
            raise Exception(f"Invalid path: {source}")
        self.img_paths: list[str] = [
            img_path
            for img_path in img_paths
            if img_path.split(".")[-1].lower() in IMG_FORMATS
        ]

        # Switch model to deploy status
        self.model_switch(self.model, self.img_size)

    def model_switch(self, model, img_size) -> None:
        """Model switch to deploy status"""
        from peekingduck.nodes.model.yolov6_core.layers.common import RepVGGBlock

        for layer in model.modules():
            if isinstance(layer, RepVGGBlock):
                layer.switch_to_deploy()

        LOGGER.info("Switch model to deploy modality.")

    def infer(
        self,
        image,
        conf_thres,
        iou_thres,
        classes,
        agnostic_nms,
        max_det,
        save_dir,
        save_txt,
        save_img,
        hide_labels,
        hide_conf,
    ) -> None:
        """Model Inference and results visualization"""

        for img_path in tqdm(self.img_paths):
            img, img_src = self.precess_image(
                img_path, self.img_size, self.stride, self.half
            )
            img = img.to(self.device)
            if len(img.shape) == 3:
                img = img[None]
                # expand for batch dim

            pred_results = self.model(img)
            det = non_max_suppression(
                pred_results,
                conf_thres,
                iou_thres,
                classes,
                agnostic_nms,
                max_det=max_det,
            )[0]

            save_path: str = osp.join(save_dir, osp.basename(img_path))  # im.jpg
            txt_path: str = osp.join(
                save_dir, "labels", osp.splitext(osp.basename(img_path))[0]
            )

            gn = torch.tensor(img_src.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            img_ori: str = img_src

            # check image and font
            assert (
                img_ori.data.contiguous
            ), "Image needs to be contiguous. Please apply to input images with np.ascontiguousarray(im)."
            self.font_check()
            LOGGER.info(
                f"img: {img.shape}\nimg_src:{img_src.shape}\nimg: {image.shape}\npred_results: {pred_results.shape}"
            )

            if len(det):
                det[:, :4] = self.rescale(
                    img.shape[2:], det[:, :4], img_src.shape
                ).round()
                LOGGER.info(det[:, :4], det[:, 4])

                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (
                            (self.box_convert(torch.tensor(xyxy).view(1, 4)) / gn)
                            .view(-1)
                            .tolist()
                        )  # normalized xywh
                        line = (cls, *xywh, conf)
                        with open(txt_path + ".txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_img:
                        class_num: int = int(cls)  # integer class
                        label = (
                            None
                            if hide_labels
                            else (
                                self.class_names[class_num]
                                if hide_conf
                                else f"{self.class_names[class_num]} {conf:.2f}"
                            )
                        )

                        self.plot_box_and_label(
                            img_ori,
                            max(round(sum(img_ori.shape) / 2 * 0.003), 2),
                            xyxy,
                            label,
                            color=self.generate_colors(class_num, True),
                        )

                img_src = np.asarray(img_ori)

                # Save results (image with detections)
                if save_img:
                    cv2.imwrite(save_path, img_src)

    # @profile(stream=fp)
    @torch.no_grad()
    def predict_object_bbox_from_image(
        self,
        image,
        conf_thres,
        iou_thres,
        classes,
        agnostic_nms,
        max_det,
    ):
        """Model Inference and results visualization"""
        img, img_src = self.precess_image(image, image.shape[1], self.stride, self.half)
        img = img.to(self.device)
        if len(img.shape) == 3:
            img = img[None]
            # expand for batch dim

        pred_results = self.model(img)
        det = non_max_suppression(
            pred_results,
            conf_thres,
            iou_thres,
            classes,
            agnostic_nms,
            max_det=max_det,
        )[0]

        img_ori = img_src

        # check image and font
        assert (
            img_ori.data.contiguous
        ), "Image needs to be contiguous. Please apply to input images with np.ascontiguousarray(im)."

        bboxes = []
        labels = []
        scores = []

        if len(det):
            det[:, :4] = self.rescale(img.shape[2:], det[:, :4], img_src.shape).round()
            # print(det[:, :4], det[:, 4])

            for *xyxy, conf, cls in reversed(det):
                class_num: int = int(cls)  # integer class
                bboxes.append(
                    xyxy2xyxyn(
                        torch.tensor(xyxy),
                        height=img[0, 0].shape[0],
                        width=img[0, 0].shape[1],
                    )
                )
                labels.append(f"{self.class_names[class_num]} {conf:.2f}")
                scores.append(float(conf))

        return bboxes, labels, scores

    @staticmethod
    def precess_image(img_src, img_size, stride, half):
        """Process image before image inference."""
        image = letterbox(img_src, img_size, stride=stride)[0]

        # Convert
        image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        image = torch.from_numpy(np.ascontiguousarray(image))
        image = image.half() if half else image.float()  # uint8 to fp16/32
        image /= 255  # 0 - 255 to 0.0 - 1.0

        return image, img_src

    @staticmethod
    def rescale(ori_shape, boxes, target_shape):
        """Rescale the output to the original image shape"""
        ratio = min(ori_shape[0] / target_shape[0], ori_shape[1] / target_shape[1])
        padding = (ori_shape[1] - target_shape[1] * ratio) / 2, (
            ori_shape[0] - target_shape[0] * ratio
        ) / 2

        boxes[:, [0, 2]] -= padding[0]
        boxes[:, [1, 3]] -= padding[1]
        boxes[:, :4] /= ratio

        boxes[:, 0].clamp_(0, target_shape[1])  # x1
        boxes[:, 1].clamp_(0, target_shape[0])  # y1
        boxes[:, 2].clamp_(0, target_shape[1])  # x2
        boxes[:, 3].clamp_(0, target_shape[0])  # y2

        return boxes

    def check_img_size(self, img_size, s=32, floor=0):
        """Make sure image size is a multiple of stride s in each dimension, and return a new shape list of image."""
        if isinstance(img_size, int):  # integer i.e. img_size=640
            new_size = max(self.make_divisible(img_size, int(s)), floor)
        elif isinstance(img_size, list):  # list i.e. img_size=[640, 480]
            new_size = [max(self.make_divisible(x, int(s)), floor) for x in img_size]
        else:
            raise Exception(f"Unsupported type of img_size: {type(img_size)}")

        if new_size != img_size:
            LOGGER.warn(
                f"WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}"
            )
        return new_size if isinstance(img_size, list) else [new_size] * 2

    def make_divisible(self, x, divisor):
        # Upward revision the value x to make it evenly divisible by the divisor.
        return math.ceil(x / divisor) * divisor

    @staticmethod
    def plot_box_and_label(
        image, lw, box, label="", color=(128, 128, 128), txt_color=(255, 255, 255)
    ):
        # Add one xyxy box to image with label
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
        if label:
            tf = max(lw - 1, 1)  # font thickness
            w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[
                0
            ]  # text width, height
            outside = p1[1] - h - 3 >= 0  # label fits outside box
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(
                image,
                label,
                (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                0,
                lw / 3,
                txt_color,
                thickness=tf,
                lineType=cv2.LINE_AA,
            )

    @staticmethod
    def font_check(
        font="./peekingduck/nodes/model/yolov6_core/utils/Arial.ttf", size=10
    ):
        # Return a PIL TrueType Font, downloading to CONFIG_DIR if necessary
        assert osp.exists(font), f"font path not exists: {font}"
        try:
            return ImageFont.truetype(str(font) if font.exists() else font.name, size)
        except Exception as e:  # download if missing
            return ImageFont.truetype(str(font), size)

    @staticmethod
    def box_convert(x):
        # Convert boxes with shape [n, 4] from [x1, y1, x2, y2] to [x, y, w, h] where x1y1=top-left, x2y2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
        y[:, 2] = x[:, 2] - x[:, 0]  # width
        y[:, 3] = x[:, 3] - x[:, 1]  # height
        return y

    @staticmethod
    def generate_colors(i, bgr=False):
        hex = (
            "FF3838",
            "FF9D97",
            "FF701F",
            "FFB21D",
            "CFD231",
            "48F90A",
            "92CC17",
            "3DDB86",
            "1A9334",
            "00D4BB",
            "2C99A8",
            "00C2FF",
            "344593",
            "6473FF",
            "0018EC",
            "8438FF",
            "520085",
            "CB38FF",
            "FF95C8",
            "FF37C7",
        )
        palette = []
        for iter in hex:
            h = "#" + iter
            palette.append(tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4)))
        num = len(palette)
        color = palette[int(i) % num]
        return (color[2], color[1], color[0]) if bgr else color
