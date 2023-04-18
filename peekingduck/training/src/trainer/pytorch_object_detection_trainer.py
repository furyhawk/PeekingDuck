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

"""pytorch trainer"""

import contextlib
import io
import itertools
import tempfile
import logging
import os
import time
from typing import Any, DefaultDict, Dict, List, Optional, Union
from collections import defaultdict
import cv2
import json

from albumentations.augmentations.transforms import Normalize
from omegaconf import DictConfig
from hydra.utils import instantiate
from pycocotools.coco import COCO
from tqdm.auto import tqdm
import numpy as np
from tabulate import tabulate
import torch  # pylint: disable=consider-using-from-import
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from configs import LOGGER_NAME

from src.data.data_adapter import DataAdapter
from src.optimizers.schedules import OptimizerSchedules
from src.losses.adapter import LossAdapter
from src.optimizers.adapter import OptimizersAdapter
from src.callbacks.base import init_callbacks
from src.callbacks.events import EVENTS
from src.model.pytorch_base import PTModel
from src.model.yoloxv1.boxes import postprocess, xyxy2xywh
from src.model.yoloxv1.visualize import vis
from src.metrics.pytorch_metrics import PytorchMetrics
from src.utils.general_utils import free_gpu_memory, time_synchronized  # , init_logger
from src.utils.pt_model_utils import set_trainable_layers, unfreeze_all_params

logger: logging.Logger = logging.getLogger(LOGGER_NAME)  # pylint: disable=invalid-name


def get_sigmoid_softmax(
    trainer_config: DictConfig,
) -> Union[torch.nn.Sigmoid, torch.nn.Softmax]:
    """Get the sigmoid or softmax function depending on loss function."""
    assert trainer_config.criterion_params.train_criterion in [
        "BCEWithLogitsLoss",
        "CrossEntropyLoss",
    ], f"Unsupported loss function {trainer_config.criterion_params.train_criterion}"

    if trainer_config.criterion_params.train_criterion == "BCEWithLogitsLoss":
        loss_func = getattr(torch.nn, "Sigmoid")()

    if trainer_config.criterion_params.train_criterion == "CrossEntropyLoss":
        loss_func = getattr(torch.nn, "Softmax")(dim=1)

    return loss_func


COCO_CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


def per_class_AR_table(
    coco_eval, class_names=COCO_CLASSES, headers=["class", "AR"], colums=6
):
    per_class_AR = {}
    recalls = coco_eval.eval["recall"]
    # dimension of recalls: [TxKxAxM]
    # recall has dims (iou, cls, area range, max dets)
    assert len(class_names) == recalls.shape[1]

    for idx, name in enumerate(class_names):
        recall = recalls[:, idx, 0, -1]
        recall = recall[recall > -1]
        ar = np.mean(recall) if recall.size else float("nan")
        per_class_AR[name] = float(ar * 100)

    num_cols = min(colums, len(per_class_AR) * len(headers))
    result_pair = [x for pair in per_class_AR.items() for x in pair]
    row_pair = itertools.zip_longest(
        *[result_pair[i::num_cols] for i in range(num_cols)]
    )
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair,
        tablefmt="pipe",
        floatfmt=".3f",
        headers=table_headers,
        numalign="left",
    )
    return table


def per_class_AP_table(
    coco_eval, class_names=COCO_CLASSES, headers=["class", "AP"], colums=6
):
    per_class_AP = {}
    precisions = coco_eval.eval["precision"]
    # dimension of precisions: [TxRxKxAxM]
    # precision has dims (iou, recall, cls, area range, max dets)
    assert len(class_names) == precisions.shape[2]

    for idx, name in enumerate(class_names):
        # area range index 0: all area ranges
        # max dets index -1: typically 100 per image
        precision = precisions[:, :, idx, 0, -1]
        precision = precision[precision > -1]
        ap = np.mean(precision) if precision.size else float("nan")
        per_class_AP[name] = float(ap * 100)

    num_cols = min(colums, len(per_class_AP) * len(headers))
    result_pair = [x for pair in per_class_AP.items() for x in pair]
    row_pair = itertools.zip_longest(
        *[result_pair[i::num_cols] for i in range(num_cols)]
    )
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair,
        tablefmt="pipe",
        floatfmt=".3f",
        headers=table_headers,
        numalign="left",
    )
    return table


# pylint: disable=too-many-instance-attributes,too-many-arguments,logging-fstring-interpolation
class PytorchTrainer:
    """Object used to facilitate training."""

    def __init__(self, framework: str = "pytorch") -> None:
        """Initialize the trainer."""
        self.framework: str = framework
        self.device: str = "cpu"

        self.trainer_config: DictConfig
        self.model_config: DictConfig
        self.callbacks_config: DictConfig
        self.metrics_config: DictConfig

        self.callbacks: list = []
        self.metrics: MetricCollection
        self.model: PTModel
        # self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        self.optimizer: torch.optim.Optimizer
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None

        self.train_params: Dict[str, Any]
        self.model_artifacts_dir: str
        self.monitored_metric: Any
        self.best_val_score: Any
        self.best_valid_loss: Any

        self.train_loader: Any
        self.validation_loader: Any

        self.stop_training: bool = False
        self.history: DefaultDict[Any, List] = defaultdict(list)
        self.epochs: int
        self.current_epoch: int
        self.current_fold: int = 0
        self.epoch_dict: Dict = {}
        self.valid_elapsed_time: str = ""
        self.train_elapsed_time: str = ""
        self.per_class_AP = True
        self.per_class_AR = True

    def setup(
        self,
        trainer_config: DictConfig,
        model_config: DictConfig,
        callbacks_config: DictConfig,
        metrics_config: DictConfig,
        data_config: DictConfig,
        device: str = "cpu",
    ) -> None:
        """Called when the trainer begins."""
        # init variables
        self.trainer_config = trainer_config[self.framework]
        self.model_config = model_config[self.framework]
        self.callbacks_config = callbacks_config[self.framework]
        self.metrics_config = metrics_config[self.framework]
        self.img_size = data_config.dataset.image_size
        self.num_classes = data_config.dataset.num_classes
        self.cls_names = list(data_config.dataset.class_name_to_id.keys())
        self.class_ids = {y: x for x, y in data_config.dataset.class_name_to_id.items()}
        self.train_params = self.trainer_config.global_train_params
        self.model_artifacts_dir = self.trainer_config.stores.model_artifacts_dir
        self.device = device
        self.epoch_dict["train"] = {}
        self.epoch_dict["validation"] = {}
        self.best_valid_loss = np.inf

        # init callbacks
        self.callbacks = init_callbacks(callbacks_config[self.framework])

        # init metrics collection
        self.metrics = PytorchMetrics.get_metrics(
            task=data_config.dataset.classification_type,
            num_classes=data_config.dataset.num_classes,
            metric_list=metrics_config[self.framework],
        )

        # create model
        torch.manual_seed(self.train_params.manual_seed)
        self.model = instantiate(
            config=self.model_config.model_type,
            cfg=self.model_config,
            _recursive_=False,
        ).to(self.device)

        # init_optimizer
        self.optimizer = OptimizersAdapter.get_pytorch_optimizer(
            model=self.model,
            optimizer=self.trainer_config.optimizer_params.optimizer,
            optimizer_params=self.trainer_config.optimizer_params.optimizer_params,
        )

        # scheduler
        if not self.trainer_config.scheduler_params.scheduler is None:
            self.scheduler = OptimizerSchedules.get_pytorch_scheduler(
                optimizer=self.optimizer,
                scheduler=self.trainer_config.scheduler_params.scheduler,
                parameters=self.trainer_config.scheduler_params.scheduler_params,
            )

        # Metric to optimize, either min or max.
        self.monitored_metric = self.train_params.monitored_metric
        self.best_val_score = (
            -np.inf if self.monitored_metric["mode"] == "max" else np.inf
        )

        self._invoke_callbacks(EVENTS.TRAINER_START.value)

    def _set_dataloaders(
        self,
        train_dl: DataLoader,
        validation_dl: DataLoader,
    ) -> None:
        """Initialise Dataloader Variables"""
        self.train_loader = train_dl
        self.validation_loader = validation_dl

    def _train_setup(self, inputs: torch.Tensor) -> None:
        self._invoke_callbacks(EVENTS.TRAINER_START.value)
        self.train_summary(inputs)

    def train_summary(self, inputs: torch.Tensor, finetune: bool = False) -> None:
        """show model layer details"""
        # if not finetune:
        #     logger.info(f"Model Layer Details:\n{self.model}")
        # show model summary
        logger.info("\n\nModel Summary:\n")
        # device parameter required for MPS,
        # otherwise the torchvision will change the model back to cpu
        # reference: https://github.com/TylerYep/torchinfo
        self.model.model_summary(inputs.shape, device=self.device)

    def _train_teardown(self) -> None:
        free_gpu_memory(
            self.optimizer,
            self.scheduler,
            # self.epoch_dict["validation"]["valid_trues"],
            # self.epoch_dict["validation"]["valid_logits"],
            # self.epoch_dict["validation"]["valid_preds"],
            # self.epoch_dict["validation"]["valid_probs"],
        )
        self._invoke_callbacks(EVENTS.TRAINER_END.value)

    def _update_epochs(self, mode: str) -> None:
        """
        Update the number of epochs based on the mode of training.
        The available options are "train", "debug" and "fine_tune".
        """
        mode_dict = {
            "train": self.train_params.epochs,
            "debug": self.train_params.debug_epochs,
            "fine-tune": self.train_params.fine_tune_epochs,
        }
        self.epochs = mode_dict.get(mode, None)
        if self.epochs is None:
            raise KeyError(f"Key '{mode}' is not valid")

    def _run_epochs(self) -> None:
        # self.epochs = self.train_params.epochs
        # if self.train_params.debug:
        #     self.epochs = self.train_params.debug_epochs

        # implement
        for epoch in range(1, self.epochs + 1):
            self.current_epoch = epoch
            self._run_train_epoch(self.train_loader)
            self._run_validation_epoch(self.validation_loader)

            if self.stop_training:  # from early stopping
                break  # Early Stopping

            if self.scheduler is not None:
                # Special Case for ReduceLROnPlateau
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(self.monitored_metric["metric_score"])
                else:
                    self.scheduler.step()

            self.epoch_dict["train"]["epoch"] = self.current_epoch
            self.epoch_dict["validation"]["epoch"] = self.current_epoch

    def _run_train_epoch(self, train_loader: DataLoader) -> None:
        """Train one epoch of the model."""
        self._invoke_callbacks(EVENTS.TRAIN_EPOCH_START.value)

        self.curr_lr = LossAdapter.get_lr(self.optimizer)
        # set to train mode
        self.model.train()

        train_bar = tqdm(train_loader)
        train_trues: List[torch.Tensor] = []
        # train_probs: List[torch.Tensor] = []

        self._invoke_callbacks(EVENTS.TRAIN_LOADER_START.value)
        # Iterate over train batches
        for _, batch in enumerate(train_bar, start=1):
            self._invoke_callbacks(EVENTS.TRAIN_BATCH_START.value)

            # unpack - note that if BCEWithLogitsLoss, dataset should do view(-1,1) and not here.
            inputs, targets, _, _ = batch
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # reset gradients
            self.optimizer.zero_grad()
            # Forward pass logits
            # with torch.cuda.amp.autocast(enabled=self.amp_training):
            logits = self.model(inputs, targets)

            loss = logits["total_loss"]
            loss.backward()
            self.optimizer.step()
            # self.scaler.update()

            # Compute the loss metrics and its gradients
            self.epoch_dict["train"]["batch_loss"] = loss.item()

            self._invoke_callbacks(EVENTS.TRAIN_BATCH_END.value)

            train_trues.extend(targets.cpu())
            # train_probs.extend(y_train_prob.cpu())

        if logits is not None:
            loss_str = ", ".join([f"{k}: {v:.1f}" for k, v in logits.items()])
            logger.info(loss_str)

        self._invoke_callbacks(EVENTS.TRAIN_LOADER_END.value)
        self._invoke_callbacks(EVENTS.TRAIN_EPOCH_END.value)

    # pylint: disable=too-many-locals
    def _run_validation_epoch(self, validation_loader: DataLoader) -> None:
        """Validate the model on the validation set for one epoch.
        Args:
            validation_loader (torch.utils.data.DataLoader): The validation set dataloader.
        Returns:
            Dict[str, np.ndarray]:
                valid_loss (float): The validation loss for each epoch.
                valid_trues (np.ndarray): The ground truth labels for each validation set.
                                            shape = (num_samples, 1)
                valid_logits (np.ndarray): The logits for each validation set.
                                            shape = (num_samples, num_classes)
                valid_preds (np.ndarray): The predicted labels for each validation set.
                                            shape = (num_samples, 1)
                valid_probs (np.ndarray): The predicted probabilities for each validation set.
                                            shape = (num_samples, num_classes)
        """
        self._invoke_callbacks(EVENTS.VALID_EPOCH_START.value)
        self.model.eval()  # set to eval mode
        valid_bar = tqdm(validation_loader)
        valid_trues: List[torch.Tensor] = []
        valid_logits: List[torch.Tensor] = []

        images_dt = []
        outputs = []

        inference_time = 0
        nms_time = 0
        n_samples = max(len(validation_loader) - 1, 1)

        self._invoke_callbacks(EVENTS.VALID_LOADER_START.value)
        std = torch.tensor([0.24703225141799082, 0.24348516474564, 0.26158783926049628])
        mean = torch.tensor(
            [0.4913997551666284, 0.48215855929893703, 0.4465309133731618]
        )
        unnormalize = Normalize(
            (-mean / std).tolist(),
            (1.0 / std).tolist(),
            always_apply=True,
            max_pixel_value=1.0,
        )

        data_list = []
        output_data = defaultdict()

        for cur_iter, batch in enumerate(valid_bar, start=1):
            with torch.no_grad():
                self._invoke_callbacks(EVENTS.VALID_BATCH_START.value)

                # unpack
                inputs, targets, info_imgs, ids = batch

                heights, widths = info_imgs
                for index, image, height, width in zip(ids, inputs, heights, widths):
                    img_info = {"id": index.item()}
                    img_info["height"] = height.item()
                    img_info["width"] = width.item()
                    img = unnormalize(
                        image=image.detach().numpy().transpose((1, 2, 0))
                    )["image"]
                    img_info["raw_img"] = (img * 255).astype(np.uint8)
                    # img, ratio = preproc(img, (height, width))
                    img_info["ratio"] = 1.0
                    images_dt.append(img_info)

                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                # skip the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(validation_loader) - 1
                if is_time_record:
                    start = time.time()

                self.optimizer.zero_grad()  # reset gradients
                logits = self.model(inputs)  # Forward pass logits

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

                self._invoke_callbacks(EVENTS.VALID_BATCH_END.value)
                # For OOF score and other computation.
                valid_trues.extend(targets.cpu())
                valid_logits.extend(logits.cpu())

                output = postprocess(logits, self.num_classes)
                outputs.extend(output)

                if is_time_record:
                    nms_end = time_synchronized()
                    nms_time += nms_end - infer_end

            data_list_elem, image_wise_data = self.convert_to_coco_format(
                outputs, info_imgs, ids, return_outputs=True
            )
            data_list.extend(data_list_elem)
            output_data.update(image_wise_data)

        # print(f"_run_validation_epoch_outputs{len(outputs)}images_dt{len(images_dt)}")

        # statistics = torch.Tensor([inference_time, nms_time, n_samples])
        # eval_results = self.evaluate_prediction(data_list, statistics)
        # print(f"eval_results{eval_results}")

        for img_info, output in zip(images_dt, outputs):
            result_image = self.visual(output, img_info)
            # if output is not None:
            #     print(f"output{output.shape}raw_img{img_info['raw_img'].shape}")
            save_folder = os.path.join(
                "YOLOX_outputs", time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
            )
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(
                save_folder, os.path.basename(".".join([str(img_info["id"]), "png"]))
            )
            logger.info(f"Saving detection result in {save_file_name}")

            # print(f"imm{result_image.shape}")
            # result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(save_file_name, result_image)

        (
            valid_trues_tensor,
            valid_logits_tensor,
        ) = (
            torch.vstack(tensors=valid_trues),
            torch.vstack(tensors=valid_logits),
        )
        # print(f"_run_validation_epoch_valid_logits{valid_logits_tensor.shape}")
        # print(f"_run_validation_epoch_valid_trues{valid_trues_tensor.shape}")
        # np.savetxt("postprocess.txt", outputs[0])
        # self.epoch_dict["validation"][
        #     "metrics"
        # ] = PytorchMetrics.get_classification_metrics(
        #     self.metrics,
        #     valid_trues_tensor,
        #     valid_probs_tensor,
        #     "val",
        # )

        self._invoke_callbacks(EVENTS.VALID_LOADER_END.value)


        self._invoke_callbacks(EVENTS.VALID_EPOCH_END.value)

    def convert_to_coco_format(
        self, outputs, info_imgs, ids, return_outputs: bool = False
    ):
        """
        Convert the outputs of the model to COCO format.
        Args:
            outputs: list of tensors, each tensor is the output of the model.
            info_imgs: list of tensors, each tensor is the info of the image.
            ids: list of tensors, each tensor is the id of the image.
            return_outputs: if True, return the outputs of the model.
        Returns:
            data_list: list of dict, each dict is the data of one image.
            image_wise_data: dict, each key is the id of the image,
                and the value is the data of the image.
        """

        data_list = []
        image_wise_data = defaultdict(dict)
        for output, img_h, img_w, img_id in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                continue
            # output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(self.img_size / float(img_h), self.img_size / float(img_w))
            bboxes /= scale
            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]

            image_wise_data.update(
                {
                    int(img_id): {
                        "bboxes": [box.numpy().tolist() for box in bboxes],
                        "scores": [score.numpy().item() for score in scores],
                        "categories": [
                            self.class_ids[int(cls[ind])]
                            for ind in range(bboxes.shape[0])
                        ],
                    }
                }
            )

            bboxes = xyxy2xywh(bboxes)

            for ind in range(bboxes.shape[0]):
                label = self.class_ids[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)

        if return_outputs:
            return data_list, image_wise_data
        return data_list

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.numpy()

        # preprocessing: resize
        bboxes = output[:, 0:4] / ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]
        print(scores)
        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res

    def _invoke_callbacks(self, event_name: str) -> None:
        """Invoke the callbacks."""
        for callback in self.callbacks:
            try:
                getattr(callback, event_name)(self)
            except NotImplementedError:
                pass

    def train(
        self,
        train_loader: DataAdapter,
        validation_loader: DataAdapter,
    ) -> Dict[str, Any]:
        """Fit the model and returns the history object."""
        self._set_dataloaders(train_dl=train_loader, validation_dl=validation_loader)
        inputs, _, _, _ = next(iter(train_loader))
        self._train_setup(inputs)  # startup
        if self.train_params.debug:
            self._update_epochs("debug")
        else:
            self._update_epochs("train")

        # check for correct fine-tune setting before start training
        assert isinstance(
            self.model_config.fine_tune, bool
        ), f"Unknown fine_tune setting '{self.model_config.fine_tune}'"

        self._run_epochs()

        # fine-tuning
        if self.model_config.fine_tune:
            if not self.train_params.debug:  # update epochs only when not in debug mode
                self._update_epochs("fine-tune")
            self._fine_tune(inputs)
        self._train_teardown()  # shutdown
        return self.history

    def _fine_tune(self, inputs: torch.Tensor) -> None:
        # update the number of epochs as fine_tune
        logger.info("\n\nUnfreezing parameters, please wait...\n")

        if self.model_config.fine_tune_all:
            unfreeze_all_params(self.model.model)
        else:
            # set fine-tune layers
            set_trainable_layers(self.model.model, self.model_config.fine_tune_modules)
        # need to re-init optimizer to update the newly unfrozen parameters
        self.optimizer = OptimizersAdapter.get_pytorch_optimizer(
            model=self.model,
            optimizer=self.trainer_config.optimizer_params.optimizer,
            optimizer_params=self.trainer_config.optimizer_params.finetune_params,
        )

        logger.info("\n\nModel Summary for fine-tuning:\n")
        self.train_summary(inputs, finetune=True)

        # run epoch
        logger.info("\n\nStart fine-tuning:\n")
        self._run_epochs()

    def evaluate_prediction(self, data_dict, statistics):
        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        nms_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = (
            1000 * inference_time / (n_samples * self.validation_loader.batch_size)
        )
        a_nms_time = 1000 * nms_time / (n_samples * self.validation_loader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "NMS", "inference"],
                    [a_infer_time, a_nms_time, (a_infer_time + a_nms_time)],
                )
            ]
        )

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = COCO("data/coco128/annotations/instances_val2017.json")
            # TODO: since pycocotools can't process dict in py36, write data to json file.
            # if self.testdev:
            #     json.dump(data_dict, open("./yolox_testdev_2017.json", "w"))
            #     cocoDt = cocoGt.loadRes("./yolox_testdev_2017.json")
            # else:
            _, tmp = tempfile.mkstemp()
            json.dump(data_dict, open(tmp, "w"))
            cocoDt = cocoGt.loadRes(tmp)
            # print(cocoDt)
            try:
                from pycocotools.cocoeval import COCOeval
            except ImportError:
                from pycocotools.cocoeval import COCOeval

                logger.warning("Use standard COCOeval.")

            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            cocoEval.evaluate()
            cocoEval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue()
            cat_ids = list(cocoGt.cats.keys())
            cat_names = [cocoGt.cats[catId]["name"] for catId in sorted(cat_ids)]
            if self.per_class_AP:
                AP_table = per_class_AP_table(cocoEval, class_names=cat_names)
                info += "per class AP:\n" + AP_table + "\n"
            if self.per_class_AR:
                AR_table = per_class_AR_table(cocoEval, class_names=cat_names)
                info += "per class AR:\n" + AR_table + "\n"
            return cocoEval.stats[0], cocoEval.stats[1], info
        else:
            return 0, 0, info
