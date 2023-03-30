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

"""Detection Trainer Pipeline"""

import os
import argparse

from src.model.yolox_megvii.data import get_yolox_datadir
from src.model.yolox_megvii.exp import Exp as MyExp



def run_detection(cfg):

    assert cfg.trainer_params.ds_format in [
        "coco",
        "voc",
    ], f"Unsupported format {cfg.trainer_params.ds_format}"

    if cfg.trainer_params.ds_format == "coco":
        exp = COCO_Exp(cfg.trainer_params.coco)
    if cfg.trainer_params.ds_format == "voc":
        exp = VOC_Exp(cfg.trainer_params.voc)

    # print(cfg)
    args = argparse.Namespace(**cfg)
    trainer = exp.get_trainer(args)
    trainer.train()


class COCO_Exp(MyExp):
    def __init__(self, cfg):
        super(MyExp, self).__init__()
        self.seed = cfg.seed
        self.output_dir = cfg.output_dir
        self.print_interval = cfg.print_interval
        self.eval_interval = cfg.eval_interval
        self.dataset = cfg.dataset
        self.num_classes = cfg.num_classes
        self.depth = cfg.depth
        self.width = cfg.width
        self.act = cfg.act
        self.data_num_workers = cfg.data_num_workers
        self.input_size = cfg.input_size
        self.multiscale_range = cfg.multiscale_range
        self.data_dir = cfg.data_dir
        self.train_ann = cfg.train_ann
        self.val_ann = cfg.val_ann
        self.test_ann = cfg.test_ann
        self.mosaic_prob = cfg.mosaic_prob
        self.mixup_prob = cfg.mixup_prob
        self.hsv_prob = cfg.hsv_prob
        self.flip_prob = cfg.flip_prob
        self.degrees = cfg.degrees
        self.translate = cfg.translate
        self.mosaic_scale = cfg.mosaic_scale
        self.enable_mixup = cfg.enable_mixup
        self.mixup_scale = cfg.mixup_scale
        self.shear = cfg.shear
        self.warmup_epochs = cfg.warmup_epochs
        self.max_epoch = cfg.max_epoch
        self.warmup_lr = cfg.warmup_lr
        self.min_lr_ratio = cfg.min_lr_ratio
        self.basic_lr_per_img = cfg.basic_lr_per_img
        self.scheduler = cfg.scheduler
        self.no_aug_epochs = cfg.no_aug_epochs
        self.ema = cfg.ema
        self.weight_decay = cfg.weight_decay
        self.momentum = cfg.momentum
        self.save_history_ckpt = cfg.save_history_ckpt
        self.exp_name = cfg.exp_name
        self.test_size = cfg.test_size
        self.test_conf = cfg.test_conf
        self.nmsthre = cfg.nmsthre


    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        """
        Get dataset according to cache and cache_type parameters.
        Args:
            cache (bool): Whether to cache imgs to ram or disk.
            cache_type (str, optional): Defaults to "ram".
                "ram" : Caching imgs to ram for fast training.
                "disk": Caching imgs to disk for fast training.
        """
        from src.model.yolox_megvii.data import COCODataset, TrainTransform

        return COCODataset(
            data_dir=os.path.join(
                "/Users/sabrimansor/Desktop/CVHUB/Training Pipeline/PeekingDuck/",
                "data",
                "coco128_2",
            ),
            json_file=self.train_ann,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob
            ),
            cache=cache,
            cache_type=cache_type,
        )

    def get_eval_dataset(self, **kwargs):
        from src.model.yolox_megvii.data import COCODataset, ValTransform
        testdev = kwargs.get("testdev", False)
        legacy = kwargs.get("legacy", False)

        return COCODataset(
            data_dir=os.path.join(
                "/Users/sabrimansor/Desktop/CVHUB/Training Pipeline/PeekingDuck/",
                "data",
                "coco128_2",
            ),
            json_file=self.val_ann if not testdev else self.test_ann,
            name="val2017" if not testdev else "test2017",
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )        

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from src.model.yolox_megvii.evaluators import COCOEvaluator

        return COCOEvaluator(
            dataloader=self.get_eval_loader(batch_size, is_distributed,
                                            testdev=testdev, legacy=legacy),
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )


class VOC_Exp(MyExp):
    def __init__(self, cfg):
        super(MyExp, self).__init__()
        self.seed = cfg.seed
        self.output_dir = cfg.output_dir
        self.print_interval = cfg.print_interval
        self.eval_interval = cfg.eval_interval
        self.dataset = cfg.dataset
        self.num_classes = cfg.num_classes
        self.depth = cfg.depth
        self.width = cfg.width
        self.act = cfg.act
        self.data_num_workers = cfg.data_num_workers
        self.input_size = cfg.input_size
        self.multiscale_range = cfg.multiscale_range
        self.data_dir = cfg.data_dir
        self.train_ann = cfg.train_ann
        self.val_ann = cfg.val_ann
        self.test_ann = cfg.test_ann
        self.mosaic_prob = cfg.mosaic_prob
        self.mixup_prob = cfg.mixup_prob
        self.hsv_prob = cfg.hsv_prob
        self.flip_prob = cfg.flip_prob
        self.degrees = cfg.degrees
        self.translate = cfg.translate
        self.mosaic_scale = cfg.mosaic_scale
        self.enable_mixup = cfg.enable_mixup
        self.mixup_scale = cfg.mixup_scale
        self.shear = cfg.shear
        self.warmup_epochs = cfg.warmup_epochs
        self.max_epoch = cfg.max_epoch
        self.warmup_lr = cfg.warmup_lr
        self.min_lr_ratio = cfg.min_lr_ratio
        self.basic_lr_per_img = cfg.basic_lr_per_img
        self.scheduler = cfg.scheduler
        self.no_aug_epochs = cfg.no_aug_epochs
        self.ema = cfg.ema
        self.weight_decay = cfg.weight_decay
        self.momentum = cfg.momentum
        self.save_history_ckpt = cfg.save_history_ckpt
        self.exp_name = cfg.exp_name
        self.test_size = cfg.test_size
        self.test_conf = cfg.test_conf
        self.nmsthre = cfg.nmsthre
        

    def get_dataset(self, cache: bool, cache_type: str = "ram"):
        from src.model.yolox_megvii.data import VOCDetection, TrainTransform

        return VOCDetection(
            data_dir=os.path.join(
                "/Users/sabrimansor/Desktop/CVHUB/Training Pipeline/PeekingDuck/",
                "data",
                "VOCdevkit",
            ),
            image_sets=[
                ("2007", "trainval")
            ],  # image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob
            ),
            cache=cache,
            cache_type=cache_type,
        )

    def get_eval_dataset(self, **kwargs):
        from src.model.yolox_megvii.data import VOCDetection, ValTransform

        legacy = kwargs.get("legacy", False)

        return VOCDetection(
            data_dir=os.path.join(
                "/Users/sabrimansor/Desktop/CVHUB/Training Pipeline/PeekingDuck/",
                "data",
                "VOCdevkit",
            ),
            image_sets=[("2007", "test")],
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from src.model.yolox_megvii.evaluators import VOCEvaluator

        return VOCEvaluator(
            dataloader=self.get_eval_loader(
                batch_size, is_distributed, testdev=testdev, legacy=legacy
            ),
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
        )

        # from src.data.data_prefetcher import DataPrefetcher
        # from src.data.data_module import ObjectDetectionDataModule
        # # from src.model.yolox import YOLOX, YOLOPAFPN, YOLOXHead
        # from src.model.yolox_megvii.models.yolo_head import YOLOXHead
        # from src.model.yolox_megvii.models.yolo_pafpn import YOLOPAFPN
        # from src.model.yolox_megvii.models.yolox import YOLOX
        # from typing import List
        # from torch import nn

        # import torch
        # import torchinfo
        # import time, tqdm

        # def init_yolo(M):
        #     for m in M.modules():
        #         if isinstance(m, nn.BatchNorm2d):
        #             m.eps = 1e-3
        #             m.momentum = 0.03

        # num_classes = 80
        # # factor of model depth
        # depth = 1.00
        # # factor of model width
        # width = 1.00

        # act = "silu"

        # device = "cpu"

        # in_channels = [256, 512, 1024]
        # backbone = YOLOPAFPN(depth, width, in_channels=in_channels, act=act)
        # head = YOLOXHead(num_classes, width, in_channels=in_channels, act=act)
        # model = YOLOX(backbone, head)

        # if cfg.device == "auto":
        #     if cfg.framework == "pytorch":
        #         cfg.device = choose_torch_device()
        #         logger.info(  # pylint: disable=logging-fstring-interpolation
        #             f"Using device: {cfg.device}"
        #         )
        #     if cfg.framework == "tensorflow":
        #         logger.info(set_tensorflow_device())

        # train_loader = get_data_loader(
        #     batch_size=1,
        #     is_distributed=False,
        #     no_aug=False,
        #     cache_img=None,
        # )

        # # data_module: DataModule = ObjectDetectionDataModule(config=cfg.data_module.module, cfg=cfg.data_module)
        # # data_module.prepare_data()
        # # data_module.setup(stage="fit")
        # # train_loader = data_module.get_train_dataloader()
        # # validation_loader = data_module.get_validation_dataloader()

        # torchinfo.summary(model, input_size=[1, 3, 640, 640], device='cpu')

        # model.to(device)
        # model.train()

        # # start training
        # for i in range(20): # 20 epoch
        #     iter_start_time = time.time()

        #     # train_bar = tqdm(train_loader)
        #     train_trues: List[torch.Tensor] = []
        #     train_probs: List[torch.Tensor] = []

        #     # Iterate over train batches
        #     for i in range(10):

        #         inps = next(iter(train_loader))
        #         inputs = inps['img'].to(device, non_blocking=True)
        #         targets = inps['bboxes'].to(device, non_blocking=True)
        #         # targets.requires_grad = False

        #         outputs = model(inputs, targets)
        #         loss = outputs["total_loss"]
        #     print("Epoch: ", i, "/20 - Loss: ", loss)

        #     # inps, targets = self.prefetcher.next()
        #     # inps = inps.to(self.data_type)
        #     # targets = targets.to(self.data_type)

        #     data_end_time = time.time()

        #     # with torch.cuda.amp.autocast(enabled=self.amp_training):
        # #     outputs = model(inps, targets)

        # #     loss = outputs["total_loss"]

        # #     self.optimizer.zero_grad()
        # #     self.scaler.scale(loss).backward()
        # #     self.scaler.step(self.optimizer)
        # #     self.scaler.update()

        # #     if self.use_model_ema:
        # #         self.ema_model.update(self.model)

        # #     lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        # #     for param_group in self.optimizer.param_groups:
        # #         param_group["lr"] = lr

        # #     iter_end_time = time.time()
        # #     self.meter.update(
        # #         iter_time=iter_end_time - iter_start_time,
        # #         data_time=data_end_time - iter_start_time,
        # #         lr=lr,
        # #         **outputs,
        # #     )
        # # model

        # from src.model.yolox_megvii.data import get_yolox_datadir

        # def get_dataset(cache: bool, cache_type: str = "ram"):
        #     from src.model.yolox_megvii.data import VOCDetection, TrainTransform
        #     input_size = (224, 224)
        #     flip_prob = 0.5
        #     hsv_prob=1.0

        #     return VOCDetection(
        #         data_dir="/Users/sabrimansor/Desktop/CVHUB/Training Pipeline/PeekingDuck/data/VOCdevkit",
        #         image_sets=[('2007', 'trainval')], # image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
        #         img_size=input_size,
        #         preproc=TrainTransform(
        #             max_labels=50,
        #             flip_prob=flip_prob,
        #             hsv_prob=hsv_prob),
        #         cache=cache,
        #         cache_type=cache_type,
        #     )

        # import torch.distributed as dist
        # def get_data_loader(batch_size, is_distributed, no_aug=False, cache_img: str = None):
        #     """
        #     Get dataloader according to cache_img parameter.
        #     Args:
        #         no_aug (bool, optional): Whether to turn off mosaic data enhancement. Defaults to False.
        #         cache_img (str, optional): cache_img is equivalent to cache_type. Defaults to None.
        #             "ram" : Caching imgs to ram for fast training.
        #             "disk": Caching imgs to disk for fast training.
        #             None: Do not use cache, in this case cache_data is also None.
        #     """
        #     from src.model.yolox_megvii.data import (
        #         TrainTransform,
        #         YoloBatchSampler,
        #         DataLoader,
        #         InfiniteSampler,
        #         MosaicDetection,
        #         worker_init_reset_seed,
        #     )
        #     from src.model.yolox_megvii.utils import wait_for_the_master

        #     # if cache is True, we will create self.dataset before launch
        #     # else we will create self.dataset after launch
        #     # if self.dataset is None:
        #     #     with wait_for_the_master():
        #     #         assert cache_img is None, \
        #     #             "cache_img must be None if you didn't create self.dataset before launch"
        #     #         self.dataset = self.get_dataset(cache=False, cache_type=cache_img)
        #     dataset = get_dataset(cache=True, cache_type=False)

        #     dataset = MosaicDetection(
        #         dataset=dataset,
        #         mosaic=not no_aug,
        #         img_size=input_size,
        #         preproc=TrainTransform(
        #             max_labels=120,
        #             flip_prob=flip_prob,
        #             hsv_prob=hsv_prob),
        #         degrees=degrees,
        #         translate=translate,
        #         mosaic_scale=mosaic_scale,
        #         mixup_scale=mixup_scale,
        #         shear=shear,
        #         enable_mixup=enable_mixup,
        #         mosaic_prob=mosaic_prob,
        #         mixup_prob=mixup_prob,
        #     )

        #     if is_distributed:
        #         batch_size = batch_size // dist.get_world_size()

        #     sampler = InfiniteSampler(len(dataset), seed=0 if None else 0)

        #     batch_sampler = YoloBatchSampler(
        #         sampler=sampler,
        #         batch_size=batch_size,
        #         drop_last=False,
        #         mosaic=not no_aug,
        #     )

        #     dataloader_kwargs = {"num_workers": 4, "pin_memory": True}
        #     dataloader_kwargs["batch_sampler"] = batch_sampler

        #     # Make sure each process has different random seed, especially for 'fork' method.
        #     # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
        #     dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        #     train_loader = DataLoader(dataset, **dataloader_kwargs)

        #     return train_loader
