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
import copy
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import re

import numpy as np
from omegaconf import DictConfig

import albumentations as A
import cv2
from PIL import Image, ImageOps
import pandas as pd
from pycocotools.coco import COCO

from configs import LOGGER_NAME

from src.model.yoloxv1.data_augment import TrainTransform, ValTransform
from src.utils.general_utils import exif_size, segments2boxes
from src.config import TORCH_AVAILABLE, TF_AVAILABLE, IMG_FORMATS
from src.utils.coco import create_coco_format_json

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
        self.targets = (
            dataframe[cfg.dataset.target_col_id].values
            if stage != "test"
            else torch.ones(1)
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
        image = cv2.imread(image_path)
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

    def apply_image_transforms(self, image: Union[Tensor, Any]) -> Union[Tensor, Any]:
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
        self.targets = (
            self.dataframe[y_col].values if stage != "test" else torch.ones(1)
        )
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


def remove_useless_info(coco):
    """
    Remove useless info in coco dataset. COCO object is modified inplace.
    This function is mainly used for saving memory (save about 30% mem).
    """
    if isinstance(coco, COCO):
        dataset = coco.dataset
        dataset.pop("info", None)
        dataset.pop("licenses", None)
        for img in dataset["images"]:
            img.pop("license", None)
            img.pop("coco_url", None)
            img.pop("date_captured", None)
            img.pop("flickr_url", None)
        if "annotations" in coco.dataset:
            for anno in coco.dataset["annotations"]:
                anno.pop("segmentation", None)


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
        """

        Args:
            cfg (DictConfig):
                Configuration dictionary.
            dataframe (pd.DataFrame):
                Dataframe with the image paths and labels.
            stage (str):
                Dataset stage.
            transforms (TransformTypes):
                Transforms to apply to the image.
            **kwargs (Dict[str, Any]):
                Additional keyword arguments.
        """

        super().__init__(**kwargs)
        self.cfg: DictConfig = cfg
        self.dataframe: pd.DataFrame = dataframe
        self.stage: str = stage
        self.transforms: TransformTypes = transforms
        self.max_labels: int = int(self.cfg.dataset.max_labels)

        self.image_path = dataframe[cfg.dataset.image_path_col_name].values
        self.label_path = (
            dataframe[cfg.dataset.target_col_id].values if stage != "test" else None
        )
        self.coco = (
            create_coco_format_json(self.image_path, self.label_path)
            if stage != "test"
            else None
        )

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.dataframe.index)

    def __getitem__(self, index: int) -> Union[Tuple, Any]:
        """Generate one batch of data

        Args:
            index (int): index of the image to return.

        Returns:
            (Dict):
                Outputs dictionary with the keys `image`, `labels`,
                `img_info`, `index`.
        """

        assert self.stage in [
            "train",
            "validation",
            "debug",
            "test",
        ], f"Invalid stage {self.stage}."

        image_path: str = self.image_path[index]
        image: np.ndarray[int, np.dtype[np.generic]] = cv2.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape
        img_info = (height, width)

        # Get target for all modes except for test dataset.
        # If test, replace target with dummy ones as placeholder.
        if self.stage != "test" and self.label_path is not None:
            label_path = self.label_path[index]
            labels = self.get_labels(label_path, self.cfg.dataset.num_classes)
            image, labels = self.apply_image_transforms(image, labels)
        else:  # Test dataset
            image = self.apply_image_transforms(image)
            labels = np.zeros((self.max_labels, 5), dtype=np.float32)

        # get id
        regex = re.compile(r"\d+")
        ids = int(regex.findall(image_path)[-1])
        return image, labels, img_info, ids

    def apply_image_transforms(self, image: torch.Tensor, labels: Optional[Any] = None):
        """Apply transforms to the image.
        Note:
            This is useful for tasks such as segmentation object detection where
            targets are in the form of bounding boxes, segmentation masks etc.
        """

        if self.transforms and isinstance(self.transforms, A.Compose):
            if labels is not None:
                bboxes = [x[1:] for x in labels]
                category_ids = [x[0] for x in labels]

                transformed = self.transforms(
                    image=image, bboxes=bboxes, category_ids=category_ids
                )
                image = transformed["image"]
                bboxes = transformed["bboxes"]
                category_ids = transformed["category_ids"]

                targets_t = np.hstack((np.array(category_ids).reshape(-1, 1), bboxes))
                padded_labels = np.zeros((self.max_labels, 5))
                padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[
                    : self.max_labels
                ]
                padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
                training_data = (image, padded_labels)
            else:
                training_data = self.transforms(image=image)["image"]
        elif self.transforms and isinstance(self.transforms, T.Compose):
            training_data = (self.transforms(image), np.zeros((0, 5), dtype=np.float32))
        else:
            training_data = (
                torch.from_numpy(image).permute(2, 0, 1),
                np.zeros((0, 5), dtype=np.float32),
            )  # convert HWC to CHW
        return training_data

    def apply_target_transforms(
        self, target: torch.Tensor, dtype: torch.dtype = torch.long
    ) -> torch.Tensor:
        """Apply transforms to the target.
        Note:
            This is useful for tasks such as segmentation object detection where
            targets are in the form of bounding boxes, segmentation masks etc.
        """
        return torch.tensor(target, dtype=dtype)

    def verify_image(self, im_file: str) -> str:
        """verify images

        Args:
            im_file (str):
                Image file path.

        Returns:
            (str):
                Message.
        """
        msg = ""
        # verify images
        image = Image.open(im_file)
        image.verify()  # PIL verify
        shape = exif_size(image)  # image size
        shape = (shape[1], shape[0])  # hw
        assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
        assert (
            image.format.lower() in IMG_FORMATS
        ), f"invalid image format {image.format}"
        if image.format.lower() in ("jpg", "jpeg"):
            with open(im_file, "rb") as f:
                f.seek(-2, 2)
                if f.read() != b"\xff\xd9":  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(
                        im_file, "JPEG", subsampling=0, quality=100
                    )
                    msg = f"WARNING ⚠️ {im_file}: corrupt JPEG restored and saved"
        return msg

    def verify_label(self, lb_file: str, num_cls: int):
        """verify labels"""
        try:
            # verify labels
            if os.path.isfile(lb_file):
                with open(lb_file) as f:
                    label = [x.split() for x in f.read().strip().splitlines() if len(x)]
                    if any(len(x) > 6 for x in label):  # is segment
                        classes = np.array([x[0] for x in label], dtype=np.float32)
                        segments = [
                            np.array(x[1:], dtype=np.float32).reshape(-1, 2)
                            for x in label
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
                                - ((label[:, 1][check3] + label[:, 3][check3] / 2) - 1)
                                * 2
                            )

                        check4 = (label[:, 2] + label[:, 4] / 2) > 1
                        if (check4).any():
                            # print(f"⚠️ 2 4 > 1 {lb[:, 1:][check4]}")
                            label[:, 4][check4] = (
                                label[:, 4][check4]
                                - ((label[:, 2][check4] + label[:, 4][check4] / 2) - 1)
                                * 2
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
                        # label empty
                        label = np.zeros((0, 5), dtype=np.float32)

            else:
                # label missing
                label = np.zeros((0, 5), dtype=np.float32)

            label = label[:, :5]
            return label
        except Exception as e:
            logger.info(f"WARNING ⚠️ {lb_file}: ignoring corrupt image/label: {e}")
            # return np.zeros((0, 5), dtype=np.float32)

    def get_labels(self, label_path: str, num_cls: int):
        """get labels from label_path
        Args:
            label_path (str): label file path
            num_cls (int): number of classes
        Returns:
            targets_t (Tensor): label tensor
        """
        targets_t = self.verify_label(label_path, num_cls)
        return targets_t


class COCODataset(Dataset):
    """
    COCO dataset class.
    """

    def __init__(
        self,
        cfg: DictConfig,
        # dataframe: pd.DataFrame,
        stage: str = "train",
        transforms=None,
        **kwargs: Dict[str, Any],
    ):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
        """
        super().__init__(**kwargs)
        self.cfg: DictConfig = cfg
        self.stage: str = stage
        self.transforms = transforms
        if self.transforms is None:
            self.transforms = TrainTransform(
                max_labels=120, flip_prob=0.5, hsv_prob=1.0
            )
        self.max_labels: int = int(self.cfg.dataset.max_labels)

        self.data_dir = self.cfg.dataset.train_dir
        self.json_file = self.cfg.dataset.json_file

        self.coco = COCO(os.path.join(self.data_dir, "annotations", self.json_file))
        remove_useless_info(self.coco)
        self.ids = self.coco.getImgIds()
        self.num_imgs = len(self.ids)
        self.class_ids = sorted(self.coco.getCatIds())
        self.cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in self.cats])
        self.name = self.cfg.dataset.name
        self.img_size = (self.cfg.dataset.image_size, self.cfg.dataset.image_size)
        self.annotations = self._load_coco_annotations()

        path_filename = [
            os.path.join(self.cfg.dataset.name, anno[3]) for anno in self.annotations
        ]

    def __len__(self):
        return self.num_imgs

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width, x1 + np.max((0, obj["bbox"][2]))))
            y2 = np.min((height, y1 + np.max((0, obj["bbox"][3]))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)

        res = np.zeros((num_objs, 5))
        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))

        file_name = (
            im_ann["file_name"]
            if "file_name" in im_ann
            else "{:012}".format(id_) + ".jpg"
        )

        return (res, img_info, resized_info, file_name)

    def load_anno(self, index):
        return self.annotations[index][0]

    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def load_image(self, index):
        file_name = self.annotations[index][3]

        img_file = os.path.join(self.data_dir, self.name, file_name)

        img = cv2.imread(img_file)
        assert img is not None, f"file named {img_file} not found"

        return img

    def read_img(self, index):
        return self.load_resized_img(index)

    def pull_item(self, index):
        id_ = self.ids[index]
        label, origin_image_size, _, _ = self.annotations[index]
        img = self.read_img(index)

        return img, copy.deepcopy(label), origin_image_size, np.array([id_])

    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w.
                h, w (int): original shape of the image
            img_id (int): same as the input index. Used for evaluation.
        """
        img, target, img_info, img_id = self.pull_item(index)

        if self.transforms is not None:
            img, target = self.transforms(img, target, self.img_size)
        return img, target, img_info, img_id
