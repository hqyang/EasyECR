#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""



Date: 2023/11/22 16:26
"""
import os
from typing import Dict
from typing import List
from typing import Any
from typing import Optional
import copy
import shutil
from enum import Enum

from omegaconf import DictConfig
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy

from easyecr.ecr_data.data_structure.data_structure import EcrData
from easyecr.ecr_model.model.ecr_model import EcrModel
from easyecr.utils import log_utils

logger = log_utils.get_logger(__file__)


class PlEcrModel(EcrModel):
    """
    1. 支持训练：根据配置和数据训练模型

    2. 支持预测：根据配置加载模型，对提供的数据进行预测

    3. 支持评估：根据配置加载模型，对提供的数据进行预测，提供的数据包含label，基于预测和label进行评估

    4. 训练、评估和预测时，当基础模型不直接得到聚类结果时，会依赖一个聚类模型

    """

    def __init__(
        self,
        mode: str,
        model_dir: str,
        model_filename: str,
        trainer_parameters: Dict[str, Any],
        best_model_filename: str = "best.ckpt",
        conf: Optional[DictConfig] = None,
    ):
        """

        Args:
            mode: train, predict, evaluate
            model_dir:
            model_filename:
            trainer_parameters:
            best_model_filename: best.ckpt
            conf: 用于存储子类中需要的配置
        """
        self.mode = mode
        self.model_dir = model_dir
        self.model_filename = model_filename
        self.best_model_filename = best_model_filename
        self.conf = conf
        self.trainer_parameters = trainer_parameters
        self.model_checkpoint = os.path.join(model_dir, best_model_filename)

        self.clean_dir()

        self.tokenizer = None
        self.build_tokenizer()

        self.module: pl.LightningModule = None
        self.build_module()

    def clean_dir(self):
        """

        Returns:

        """
        if self.mode == "train":
            pass

    def build_tokenizer(self):
        raise NotImplementedError()

    def build_module(self):
        if self.mode == "train":
            self.module = self.instanciate_module()
        else:
            model_checkpoint = self.model_checkpoint
            self.module = self.load_module(model_checkpoint)

    def instanciate_module(self):
        raise NotImplementedError()

    def load_module(self, filepath: str):
        raise NotImplementedError()

    def prepare_data(self, data: EcrData, mode: str):
        """

        Args:
            data:
            mode: train, evaluate, predict

        Returns:

        """
        return None, None

    def instantiate_model_checkpoint_callback(self) -> Callback:
        callback = ModelCheckpoint(save_top_k=-1, verbose=True, dirpath=self.model_dir, filename=self.model_filename)
        return callback

    def get_all_versions(self) -> List[str]:
        """获得模型训练过程中保存的所有模型版本
        配合set_model_version，使得外层框架可以确定最优模型版本

        Returns: version用str表示
        """
        model_versions = []
        model_dir = self.model_dir
        model_filenames = os.listdir(model_dir)
        for filename in model_filenames:
            filepath = os.path.join(model_dir, filename)
            model_versions.append(filepath)
        return model_versions

    def set_model_version(self, version: int, is_best: bool = False):
        """设定当前需要使用的模型版本
        配合get_all_versions，使得外层框架可以确定最优模型版本
        Args:
            version:
            is_best:

        Returns:

        """
        self.module = self.load_module(version)
        if is_best:
            logger.info(f"original best model filepath: {version}")
            model_dir = self.model_dir
            best_model_file_name = self.best_model_filename
            final_best_model_filepath = os.path.join(model_dir, best_model_file_name)
            shutil.copy(version, final_best_model_filepath)
            logger.info(f"final best model filepath: {final_best_model_filepath}")

    def add_callbacks(self, train_data: EcrData, dev_data: EcrData) -> List[Callback]:
        """

        Args:
            train_data:
            dev_data:

        Returns:

        """
        result = []
        return result

    def train(self, train_data: EcrData, dev_data: EcrData):
        """
        1. 不使用early stopping，保存每个epoch的模型，训练完后，筛选最优模型。相比于通过callback在每个epoch遍历
        聚类模型的每种超惨，这样做的优势是，基本模型和聚类模型解耦合了，在基本模型训练完后，还可以尝试不同的聚类模型的不同配置
        Args:
            train_data:
            dev_data:

        Returns:

        """
        trainer_parameters = self.trainer_parameters
        model_checkpoint_callback = self.instantiate_model_checkpoint_callback()
        trainer_parameters["callbacks"] = [model_checkpoint_callback]
        trainer_parameters["callbacks"].extend(self.add_callbacks(train_data, dev_data))
        trainer_parameters["strategy"] = "ddp_find_unused_parameters_true"
        trainer = pl.Trainer(**trainer_parameters)

        train_dataset, train_dataloaders = self.prepare_data(train_data, mode=self.mode)
        dev_dataset, dev_dataloaders = self.prepare_data(dev_data, mode=self.mode)
        # train_dataloaders = dev_dataloaders
        trainer.fit(self.module, train_dataloaders=train_dataloaders, val_dataloaders=dev_dataloaders)
