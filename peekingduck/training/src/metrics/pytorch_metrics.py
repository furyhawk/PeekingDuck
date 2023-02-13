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

import pandas as pd
import tabulate
import torch
from torchmetrics import AUROC, Accuracy, MetricCollection, Precision, Recall
from torchmetrics.classification import MulticlassCalibrationError
from typing import List, Dict
from omegaconf import DictConfig
from src.metrics.base import MetricsAdapter


class PytorchMetrics(MetricsAdapter):
    def __init__(
        self,
        task: str = "multiclass",
        num_classes: int = 2,
        metrics: List[str] = None,
    ) -> None:
        self.metrics_collection = None
        self.num_classes = num_classes
        self.task = task
        self.metrics = metrics
        self.metricList = {}
        for metric in self.metrics:
            try:
                if type(metric) is DictConfig:
                    for mkey, mval in metric.items():
                        self.metricList[metric] = getattr(self, mkey)(mval)
                elif type(metric) is str:
                    self.metricList[metric] = getattr(self, metric)()
                else:
                    raise TypeError
            except NotImplementedError:
                raise NotImplementedError

        for metric in self.metrics:
            try:
                self.metricList[metric] = getattr(self, metric)()
            except NotImplementedError:
                raise NotImplementedError

    def accuracy(self, parameters: Dict = {}):
        return Accuracy(task=self.task, num_classes=self.num_classes, **parameters)

    def precision(self, parameters: Dict = {}):
        return Precision(
            task=self.task, num_classes=self.num_classes, average="macro", **parameters
        )

    def recall(self, parameters: Dict = {}):
        return Recall(
            task=self.task, num_classes=self.num_classes, average="macro", **parameters
        )

    def auroc(self, parameters: Dict = {}):
        return AUROC(
            task=self.task, num_classes=self.num_classes, average="macro", **parameters
        )

    def multiclass_calibration_error(self, parameters: Dict = {}):
        return MulticlassCalibrationError(num_classes=self.num_classes, **parameters)

    def get_metrics(self) -> MetricCollection:
        self.metrics_collection = MetricCollection(list(self.metricList.values()))
        return self.metrics_collection


    @staticmethod
    def get_classification_metrics(
        metrics,
        y_trues: torch.Tensor,
        y_preds: torch.Tensor,
        y_probs: torch.Tensor,
    ):
        """[summary]
        # https://ghnreigns.github.io/reighns-ml-website/supervised_learning/classification/breast_cancer_wisconsin/Stage%206%20-%20Modelling%20%28Preprocessing%20and%20Spot%20Checking%29/
        Args:
            y_trues (torch.Tensor): dtype=[torch.int64], shape=(num_samples, 1); (May be float if using BCEWithLogitsLoss)
            y_preds (torch.Tensor): dtype=[torch.int64], shape=(num_samples, 1);
            y_probs (torch.Tensor): dtype=[torch.float32], shape=(num_samples, num_classes);
            mode (str, optional): [description]. Defaults to "valid".
        """

        train_metrics = metrics.clone(prefix="train_")
        valid_metrics = metrics.clone(prefix="val_")

        # FIXME: currently train and valid give same results, since this func call takes in
        # y_trues, etc from valid_one_epoch.
        train_metrics_results = train_metrics(y_probs, y_trues.flatten())
        train_metrics_results_df = pd.DataFrame.from_dict([train_metrics_results])

        valid_metrics_results = valid_metrics(y_probs, y_trues.flatten())
        valid_metrics_results_df = pd.DataFrame.from_dict([valid_metrics_results])

        # TODO: relinquish this logging duty to a callback or for now in train_one_epoch and valid_one_epoch.
        # self.logger.info(
        #     f"\ntrain_metrics:\n{tabulate(train_metrics_results_df, headers='keys', tablefmt='psql')}\n"
        # )
        # self.logger.info(
        #     f'\nvalid_metrics:\n{tabulate(valid_metrics_results_df, headers="keys", tablefmt="psql")}\n'
        # )

        return train_metrics_results, valid_metrics_results
