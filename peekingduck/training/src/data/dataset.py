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

"""dataset"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import numpy as np
from omegaconf import DictConfig

import albumentations as A
import cv2
from PIL import Image, ImageOps
import pandas as pd
from configs import LOGGER_NAME

from src.utils.general_utils import exif_size, segments2boxes
from src.config import TORCH_AVAILABLE, TF_AVAILABLE, IMG_FORMATS

logger = logging.getLogger(LOGGER_NAME)  # pylint: disable=invalid-name

if TORCH_AVAILABLE:
    import torch
    from torch import Tensor
    from torch.utils.data import Dataset
    import torchvision.transforms as T
else:
    raise ImportError("Called a torch-specific function but torch is not installed.")

if TF_AVAILABLE:
    import tensorflow as tf
else:
    raise ImportError(
        "Called a tensorflow-specific function but tensorflow is not installed/available."
    )

TransformTypes = Optional[Union[A.Compose, T.Compose]]


class PTImageClassificationDataset(
    Dataset
):  # pylint: disable=too-many-instance-attributes, too-many-arguments
    """Template for Image Classification Dataset."""

    def __init__(
        self,
        cfg: DictConfig,
        dataframe: pd.DataFrame,
        stage: str = "train",
        transforms: TransformTypes = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        """"""

        super().__init__(**kwargs)
        self.cfg: DictConfig = cfg
        self.dataframe: pd.DataFrame = dataframe
        self.stage: str = stage
        self.transforms: TransformTypes = transforms

        self.image_path = dataframe[cfg.dataset.image_path_col_name].values
        self.label_path = (
            dataframe[cfg.dataset.target_col_id].values if stage != "test" else None
        )

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.dataframe.index)

    def __getitem__(self, index: int) -> Union[Tuple, Any]:
        """Generate one batch of data"""
        assert self.stage in [
            "train",
            "valid",
            "debug",
            "test",
        ], f"Invalid stage {self.stage}."

        image_path: str = self.image_path[index]
        image: Tensor = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.apply_image_transforms(image)

        # Get target for all modes except for test dataset.
        # If test, replace target with dummy ones as placeholder.
        target = self.targets[index] if self.stage != "test" else torch.ones(1)
        target = self.apply_target_transforms(target)
        # target = self.apply_target_transforms(target)

        if self.stage in ["train", "valid", "debug"]:
            return image, target
        # self.stage == "test"
        return image

    def apply_image_transforms(self, image: torch.Tensor) -> Tensor:
        """Apply transforms to the image."""
        if self.transforms and isinstance(self.transforms, A.Compose):
            image = self.transforms(image=image)["image"]
        elif self.transforms and isinstance(self.transforms, T.Compose):
            image = self.transforms(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1)  # convert HWC to CHW
        return image

    def apply_target_transforms(
        self, target: torch.Tensor, dtype: torch.dtype = torch.long
    ) -> torch.Tensor:
        """Apply transforms to the target.
        Note:
            This is useful for tasks such as segmentation object detection where
            targets are in the form of bounding boxes, segmentation masks etc.
        """
        return torch.tensor(target, dtype=dtype)


class TFImageClassificationDataset(
    tf.keras.utils.Sequence
):  # pylint: disable=too-many-instance-attributes, too-many-arguments, invalid-name
    """Template for Image Classification Dataset."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        stage: str = "train",
        batch_size: int = 1,
        num_classes: int = 2,
        target_size: Union[list, tuple] = (32, 32),
        shuffle: bool = False,
        transforms: TransformTypes = None,
        num_channels: int = 3,
        x_col: str = "",
        y_col: str = "",
        **kwargs: Dict[str, Any],
    ) -> None:
        """"""

        self.dataframe = dataframe
        self.stage = stage
        self.transforms = transforms
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.dim = target_size
        self.num_channels = num_channels
        self.shuffle = shuffle

        self.image_paths = self.dataframe[x_col].values
        self.targets = self.dataframe[y_col].values if stage != "test" else None
        self.kwargs = kwargs
        self._on_epoch_end()

    def __len__(self) -> int:
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index: int) -> Union[Tuple, Any]:
        """Generate one batch of data

        Args:
            index (int): index of the image to return.

        Returns:
            (Dict):
                Outputs dictionary with the keys `labels`.
        """
        assert self.stage in [
            "train",
            "valid",
            "debug",
            "test",
        ], f"Invalid stage {self.stage}."

        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Find list of IDs
        image_paths_temp: List[Any] = [self.image_paths[k] for k in indexes]

        # Generate data
        X, y = self._data_generation(image_paths_temp, indexes)  # pylint: disable=W0631

        if self.stage in ["train", "validation", "debug"]:
            return X, y
        # self.stage == "test"
        return X

    def load_image(self, image_path: str) -> Any:
        """Load image from `image_path`

        Args:
            image_path (str): image path to load.

        Returns:
            (ndarray):
                Outputs image in numpy array.
            Preprocessed numpy.array or a tf.Tensor with type float32.
            The images are converted from RGB to BGR,
            then each color channel is zero-centered with respect to
            the ImageNet dataset, without scaling.
        """
        image = cv2.imread(image_path)  # BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self.apply_image_transforms(image)
        return image

    # pylint: disable=invalid-name
    def _data_generation(
        self,
        list_ids_temp: List[Any],
        indexes: np.ndarray,
    ) -> Any:
        "Generates data containing batch_size samples"  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.num_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, image_path in enumerate(list_ids_temp):
            # Store sample
            X[i] = self.load_image(image_path)

        # Store class
        y = self.targets[indexes]

        return X, tf.keras.utils.to_categorical(y, num_classes=self.num_classes)

    def apply_image_transforms(self, image: Any) -> Any:
        """Apply transforms to the image."""
        if self.transforms and isinstance(self.transforms, A.Compose):
            image = self.transforms(image=image)["image"]
        elif self.transforms and isinstance(self.transforms, T.Compose):
            image = self.transforms(image)
        return image

    def _on_epoch_end(self) -> None:
        "Updates indexes after each epoch"
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)


class PTObjectDetectionDataset(Dataset):
    """Template for Object Detection Dataset."""

    def __init__(
        self,
        cfg: DictConfig,
        dataframe: pd.DataFrame,
        stage: str = "train",
        transforms: TransformTypes = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        """"""

        super().__init__(**kwargs)
        self.cfg: DictConfig = cfg
        self.dataframe: pd.DataFrame = dataframe
        self.stage: str = stage
        self.transforms: TransformTypes = transforms

        self.image_path = dataframe[cfg.dataset.image_path_col_name].values
        self.label_path = (
            dataframe[cfg.dataset.target_col_id].values if stage != "test" else None
        )

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.dataframe.index)

    def __getitem__(self, index: int) -> Union[Tuple, Any]:
        """Generate one batch of data"""
        assert self.stage in [
            "train",
            "valid",
            "debug",
            "test",
        ], f"Invalid stage {self.stage}."

        image_path: str = self.image_path[index]
        image: Tensor = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.apply_image_transforms(image)

        # Get target for all modes except for test dataset.
        # If test, replace target with dummy ones as placeholder.
        training_data = {"labels": []}
        if self.stage != "test":
            label_path = self.label_path[index]
            training_data = self.get_labels(label_path, self.cfg.dataset.num_classes)
            training_data["img"] = image

        return training_data

    def apply_image_transforms(self, image: torch.Tensor) -> Tensor:
        """Apply transforms to the image."""
        if self.transforms and isinstance(self.transforms, A.Compose):
            image = self.transforms(image=image)["image"]
        elif self.transforms and isinstance(self.transforms, T.Compose):
            image = self.transforms(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1)  # convert HWC to CHW
        return image

    def apply_target_transforms(
        self, target: torch.Tensor, dtype: torch.dtype = torch.long
    ) -> torch.Tensor:
        """Apply transforms to the target.
        Note:
            This is useful for tasks such as segmentation object detection where
            targets are in the form of bounding boxes, segmentation masks etc.
        """
        # self.labels = self.get_labels()
        return torch.tensor(target, dtype=dtype)

    def verify_image(self, im_file):
        msg = ""
        # verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        shape = (shape[1], shape[0])  # hw
        assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
        assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}"
        if im.format.lower() in ("jpg", "jpeg"):
            with open(im_file, "rb") as f:
                f.seek(-2, 2)
                if f.read() != b"\xff\xd9":  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(
                        im_file, "JPEG", subsampling=0, quality=100
                    )
                    msg = f"WARNING ⚠️ {im_file}: corrupt JPEG restored and saved"
        return msg

    def verify_label(self, lb_file, num_cls):
        try:
            # verify labels
            if os.path.isfile(lb_file):
                with open(lb_file) as f:
                    lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                    if any(len(x) > 6 for x in lb):  # is segment
                        classes = np.array([x[0] for x in lb], dtype=np.float32)
                        segments = [
                            np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb
                        ]  # (cls, xy1...)
                        lb = np.concatenate(
                            (classes.reshape(-1, 1), segments2boxes(segments)), 1
                        )  # (cls, xywh)
                    lb = np.array(lb, dtype=np.float32)
                    nl = len(lb)
                    if nl:
                        assert (
                            lb.shape[1] == 5
                        ), f"labels require 5 columns, {lb.shape[1]} columns detected"
                        assert (
                            lb[:, 1:] <= 1
                        ).all(), f"non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}"

                        # All labels
                        max_cls = int(lb[:, 0].max())  # max label count
                        assert max_cls <= num_cls, (
                            f"Label class {max_cls} exceeds dataset class count {num_cls}. "
                            f"Possible class labels are 0-{num_cls - 1}{lb}"
                        )
                        assert (lb >= 0).all(), f"negative label values {lb[lb < 0]}"
                        _, i = np.unique(lb, axis=0, return_index=True)
                        if len(i) < nl:  # duplicate row check
                            lb = lb[i]  # remove duplicates
                            if segments:
                                segments = [segments[x] for x in i]
                            logger.info(
                                f"WARNING ⚠️ {lb_file}: {nl - len(i)} duplicate labels removed"
                            )
                    else:
                        ne = 1  # label empty
                        lb = np.zeros((0, 5), dtype=np.float32)

            else:
                nm = 1  # label missing
                lb = np.zeros((0, 5), dtype=np.float32)

            lb = lb[:, :5]
            return lb
        except Exception as e:
            nc = 1
            logger.info(f"WARNING ⚠️ {lb_file}: ignoring corrupt image/label: {e}")

    def get_labels(self, label_path, num_cls):
        # print(num_cls)
        # y = {"labels": []}  # pylint: disable=W0631
        targets = self.verify_label(label_path, num_cls)
        # y["labels"].append(
        #     dict(
        #         cls=targets[:, 0:1],  # n, 1
        #         bboxes=targets[:, 1:],  # n, 4
        #     )
        # )
        return dict(
            cls=targets[:, 0:1],  # n, 1
            bboxes=targets[:, 1:],  # n, 4
        )
