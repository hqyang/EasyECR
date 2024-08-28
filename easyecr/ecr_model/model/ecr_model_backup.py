#!/usr/bin/env python
# -*- coding:utf-8 -*-　
"""



Date: 2023/11/22 16:26
"""
import os
from typing import Dict
from typing import Optional
import copy
import shutil

from omegaconf import DictConfig
from omegaconf import OmegaConf
import pytorch_lightning as pl
from transformers import AutoTokenizer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data.dataloader import DataLoader
import torch

from easyecr.ecr_data.data_structure.data_structure import EcrData
from easyecr.ecr_model.data.event_coreference_dataset import EventCoreferenceDataset
from easyecr.ecr_model.model.event_coreference_pl_module import EventCoreferenceModule
from easyecr.ecr_model.cluster.cluster_model import EcrClusterModel
from easyecr.ecr_evaluate.ecr_evaluate import Evaluator
from easyecr.utils import log_utils

logger = log_utils.get_logger(__file__)


class EcrModel:
    """
    1. 支持训练：根据配置和数据训练模型

    2. 支持预测：根据配置加载模型，对提供的数据进行预测

    3. 支持评估：根据配置加载模型，对提供的数据进行预测，提供的数据包含label，基于预测和label进行评估

    4. 训练、评估和预测时，当基础模型不直接得到聚类结果时，会依赖一个聚类模型

    """
    def __init__(self, conf: Optional[DictConfig],
                 cluster_model: Optional[EcrClusterModel] = None,
                 evaluator: Optional[Evaluator] = None):
        """

        :param conf: include trainer_parameters
        :param cluster_model:
        :param evaluator:
        """
        self.conf = conf
        self.cluster_model = cluster_model
        self.evaluator = evaluator

        self.tokenizer = None
        self.build_tokenizer()
        
        self.module: pl.LightningModule = None
        self.build_module()

    def build_tokenizer(self):
        pass

    def build_module(self):
        pass

    def load_module(self, filepath: str):
        pass

    def prepare_data(self, data: EcrData, train: bool):
        """

        Args:
            data:
            train:

        Returns:

        """
        return None, None

    def instanciate_callbacks(self, callbacks_config):
        result = []
        for callback_name, config in callbacks_config.items():
            if callback_name == 'EarlyStopping':
                callback = EarlyStopping(**config)
            elif callback_name == 'ModelCheckpoint':
                callback = ModelCheckpoint(**config)
            else:
                raise NotImplementedError(callback_name)
            result.append(callback)
        return result

    def get_best_metrics(self, key_and_best_metrics: Dict[str, Dict]):
        """

        Args:
            key_and_best_metrics:

        Returns:

        """
        best_key = None
        best_metrics = None
        for key, metrics in key_and_best_metrics.items():
            if best_key is None or metrics[self.conf['evaluate']['main_metric']]['f1'] \
                    > best_metrics[self.conf['evaluate']['main_metric']]['f1']:
                best_key = key
                best_metrics = metrics
        result = [best_key, best_metrics]
        return result

    def find_the_best_model(self, dev_data: EcrData):
        """

        Args:
            dev_data:

        Returns:

        """
        model_dir = self.conf['ecr_model']['trainer_parameters']['callbacks']['ModelCheckpoint']['dirpath']
        model_filenames = os.listdir(model_dir)
        best_model = None
        best_metrics = None
        best_model_filepath = None
        for filename in model_filenames:
            filepath = os.path.join(model_dir, filename)
            self.module = self.load_module(filepath)
            key_and_metrics = self.evaluate(dev_data)
            best_key_and_metrics = self.get_best_metrics(key_and_metrics)
            if best_model is None or best_key_and_metrics[1][self.conf['evaluate']['main_metric']]['f1'] \
                > best_metrics[1][self.conf['evaluate']['main_metric']]['f1']:
                best_model = self.module
                best_metrics = best_key_and_metrics
                best_model_filepath = filepath
        self.module = best_model
        logger.info(f'best_metrics: {best_metrics}')
        final_best_model_filepath = os.path.join(model_dir, 'best.ckpt')
        logger.info(f'best_model_filepath: {best_model_filepath}')
        shutil.copy(best_model_filepath, final_best_model_filepath)

    def train(self, train_data: EcrData, dev_data: EcrData):
        """
        1. 不使用early stopping，保存每个epoch的模型，训练完后，筛选最优模型。相比于通过callback在每个epoch遍历
        聚类模型的每种超惨，这样做的优势是，基本模型和聚类模型解耦合了，在基本模型训练完后，还可以尝试不同的聚类模型的不同配置
        Args:
            train_data:
            dev_data:

        Returns:

        """
        train_dataset, train_dataloaders = self.prepare_data(train_data, train=True)
        dev_dataset, dev_dataloaders = self.prepare_data(dev_data, train=True)

        trainer_parameters = copy.deepcopy(self.conf['ecr_model']['trainer_parameters'])
        trainer_parameters = OmegaConf.to_container(trainer_parameters)
        trainer_parameters['callbacks'] = self.instanciate_callbacks(trainer_parameters['callbacks'])

        trainer = pl.Trainer(**trainer_parameters)

        trainer.fit(self.module,
                    train_dataloaders=train_dataloaders,
                    val_dataloaders=dev_dataloaders
                    )
        self.find_the_best_model(dev_data)

    def evaluate(self, data: EcrData, keep_singletons: bool = False):
        """ 给mention打上标签, 并基于ground truth进行评估

        Args:
            data:
            keep_singletons:

        Returns:

        """
        data.add_event_id()
        pred_tag = 'event_id_pred'
        predicted_data = self.predict(data, pred_tag)
        all_pred_tag = data.get_mention_tags_by_prefix(pred_tag)
        result = {}
        for tag in all_pred_tag:
            tag_metrics = self.evaluator.evaluate_from_ecr_data(predicted_data, tag, keep_singletons)
            result[tag] = tag_metrics
        return result

    def inner_pred(self, trainer, module, dataloaders, dataset):
        """

        :param trainer:
        :param module:
        :param dataloaders:
        :param dataset:
        :param distance_threshold:
        :return:
        """
        predictions = trainer.predict(module, dataloaders=dataloaders)
        representations = [e['mention_representations1'] for e in predictions]
        representations = torch.cat(representations, dim=0).cpu().numpy()
        group_representations = dataset.split_representations_by_coarse_cluster_label(representations)
        return group_representations

    def predict(self, data: EcrData, output_tag: str) -> EcrData:
        """ 给mention打上标签
        1. 预测mention
        2. 预测共指

        :param data:
        :param output_tag:
        :return:
        """
        trainer = pl.Trainer(
            accelerator="gpu",
            # devices=args.devices
        )
        test_dataset, test_dataloaders = self.prepare_data(data, train=False)
        group_representations = self.inner_pred(trainer, self.module, test_dataloaders, test_dataset)
        repr_tag = 'repr'
        mention_id_and_value = {}
        for coarse_type, repres in group_representations.items():
            for item in repres:
                repre = item[0]
                mention_key = item[1]
                mention_id_and_value[mention_key] = repre
        data.add_mention_tag(repr_tag, mention_id_and_value)

        data = self.cluster_model.predict_by_repr(data, repr_tag, output_tag,
                                                  self.conf['data_parameters']['coarse_tag'])
        return data


class ContrastiveRepresentationLearningEcrModel(EcrModel):
    """
    Contrastive Representation Learning for Cross-Document Coreference Resolution of Events and Entities
    """

    def __init__(self, conf: Optional[DictConfig],
                 cluster_model: Optional['EcrModel'] = None,
                 evaluator: Optional[Evaluator] = None):
        super().__init__(conf, cluster_model=cluster_model, evaluator=evaluator)

    def build_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.conf['model_parameters']['transformer_model'])

    def build_module(self):
        if self.conf['common']['mode'] == 'train':
            self.module = EventCoreferenceModule(self.conf['model_parameters'])
        else:
            model_checkpoint = self.conf['model_parameters']['model_checkpoint']
            self.module = self.load_module(model_checkpoint)

    def load_module(self, filepath: str):
        """

        :param filepath:
        :return:
        """
        result = EventCoreferenceModule.load_from_checkpoint(filepath)
        return result

    def prepare_data(self, data: EcrData, train: bool):
        """

        Args:
            data:
            train:

        Returns:

        """
        if data is None:
            return None, None
        coarse_type = self.conf['data_parameters']['coarse_type']
        dataset = EventCoreferenceDataset(data, self.tokenizer, coarse_type, train=train)

        batch_size = self.conf['data_parameters']['batch_size']
        num_workers = self.conf['data_parameters']['num_workers']
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                collate_fn=EventCoreferenceDataset.collate_fn,
                                num_workers=num_workers)
        dataloaders = [dataloader]
        return dataset, dataloaders
