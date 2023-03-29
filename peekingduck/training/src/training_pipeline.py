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

"""Trainer Pipeline"""

import logging
import os
from typing import Any, Dict, Optional
from time import perf_counter
from numpy import dtype
from omegaconf import DictConfig
from hydra.utils import instantiate
from configs import LOGGER_NAME

from src.data.base import DataModule
from src.trainer.base import Trainer
from src.utils.general_utils import choose_torch_device, set_tensorflow_device
from src.model_analysis.weights_biases import WeightsAndBiases

logger: logging.Logger = logging.getLogger(LOGGER_NAME)  # pylint: disable=invalid-name


def init_trainer(cfg: DictConfig) -> Trainer:
    """Instantiate Trainer Pipeline"""
    trainer: Trainer = instantiate(
        cfg.trainer[cfg.framework].global_train_params.trainer, cfg.framework
    )
    trainer.setup(
        cfg.trainer,
        cfg.model,
        cfg.callbacks,
        cfg.metrics,
        cfg.data_module,
        device=cfg.device,
    )
    return trainer

def run(cfg: DictConfig) -> None:

    if cfg.use_case_testing == "classification":
        """Run the Trainer Pipeline"""
        assert cfg.framework in [
            "pytorch",
            "tensorflow",
        ], f"Unsupported framework {cfg.framework}"
        start_time: float = perf_counter()

        if cfg.device == "auto":
            if cfg.framework == "pytorch":
                cfg.device = choose_torch_device()
                logger.info(  # pylint: disable=logging-fstring-interpolation
                    f"Using device: {cfg.device}"
                )
            if cfg.framework == "tensorflow":
                logger.info(set_tensorflow_device())

        data_module: DataModule = instantiate(
            config=cfg.data_module.module,
            cfg=cfg.data_module,
        )
        data_module.prepare_data()
        data_module.setup(stage="fit")
        train_loader = data_module.get_train_dataloader()
        validation_loader = data_module.get_validation_dataloader()

        if cfg.view_only:
            trainer: Trainer = init_trainer(cfg)
            inputs, _ = next(iter(train_loader))
            trainer.train_summary(inputs)
        else:
            model_analysis: WeightsAndBiases = WeightsAndBiases(cfg.model_analysis)
            trainer = init_trainer(cfg)
            history: Dict[str, Any] = trainer.train(train_loader, validation_loader)
            model_analysis.log_history(history)

            end_time: float = perf_counter()
            run_time: str = f"Run time = {end_time - start_time:.2f} sec"
            logger.info(run_time)
            model_analysis.log({"run_time": end_time - start_time})

    elif cfg.use_case_testing == "detection":
        import argparse

        exp = Exp()
        
        kwargs = {}
        kwargs['batch_size']=64
        kwargs['resume']=False
        kwargs['ckpt']='peekingduck/training/weights/yolox_s.pth'
        kwargs['fp16']=None
        kwargs['occupy']=False
        kwargs['name']="yolox_s"
        kwargs['experiment_name']="fashion_dataset"
        kwargs['cache']=None
        kwargs['logger']='tensorboard'

        print(kwargs)
        args = argparse.Namespace(**kwargs)
        print(args.fp16)
        trainer = exp.get_trainer(args)
        trainer.train()

from src.model.yolox_megvii.data import get_yolox_datadir
from src.model.yolox_megvii.exp import Exp as MyExp
class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 11
        self.max_epoch=3
        self.depth = 0.33
        self.width = 0.50
        self.warmup_epochs = 1

        # ---------- transform config ------------ #
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

    def get_dataset(self, cache: bool, cache_type: str = "ram"):
        from src.model.yolox_megvii.data import VOCDetection, TrainTransform

        return VOCDetection(
            data_dir=os.path.join("/Users/sabrimansor/Desktop/CVHUB/Training Pipeline/PeekingDuck/", "data", "VOCdevkit"),
            image_sets=[('2007', 'trainval')], # image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob),
            cache=cache,
            cache_type=cache_type,
        )

    def get_eval_dataset(self, **kwargs):
        from src.model.yolox_megvii.data import VOCDetection, ValTransform
        legacy = kwargs.get("legacy", False)

        return VOCDetection(
            data_dir=os.path.join("/Users/sabrimansor/Desktop/CVHUB/Training Pipeline/PeekingDuck/", "data", "VOCdevkit"),
            image_sets=[('2007', 'test')],
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from src.model.yolox_megvii.evaluators import VOCEvaluator

        return VOCEvaluator(
            dataloader=self.get_eval_loader(batch_size, is_distributed,
                                            testdev=testdev, legacy=legacy),
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
