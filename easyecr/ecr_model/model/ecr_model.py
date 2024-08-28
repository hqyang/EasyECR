#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""



Date: 2023/11/22 16:26
"""
from typing import List
from typing import Dict
from enum import Enum

from easyecr.ecr_data.data_structure.data_structure import EcrData
from easyecr.ecr_data.data_structure.data_structure import Mention
from easyecr.utils import log_utils
from easyecr.ecr_model.ecr_tagger.ecr_tagger import EcrTagger

logger = log_utils.get_logger(__file__)


class EcrModel(EcrTagger):
    def train(self, train_data: EcrData, dev_data: EcrData):
        """

        Args:
            train_data:
            dev_data:

        Returns:

        """
        raise NotImplementedError()

    def get_all_versions(self) -> List[str]:
        """获得模型训练过程中保存的所有模型版本
        配合set_model_version，使得外层框架可以确定最优模型版本

        Returns: version用str表示
        """
        raise NotImplementedError()

    def set_model_version(self, version: int, is_best: bool = False):
        """设定当前需要使用的模型版本
        配合get_all_versions，使得外层框架可以确定最优模型版本
        Args:
            version:
            is_best:

        Returns:

        """
        raise NotImplementedError()

    def predict_reprs(self, data: EcrData) -> Dict[str, List[float]]:
        """

        Args:
            data:

        Returns:

        """
        raise NotImplementedError()

    def predict_distances(self, data: EcrData) -> Dict[str, Dict[str, float]]:
        """

        Args:
            data:

        Returns:

        """
        raise NotImplementedError()

    def predict_lemma(self):
        raise NotImplementedError()

    def get_predict_type(self) -> str:
        """

        Returns: repr, distance

        """
        raise NotImplementedError()

    def predict(self, data: EcrData, output_tag: str) -> EcrData:
        """


        Args:
            data:
            output_tag:

        Returns:

        """
        predict_type = self.get_predict_type()
        if predict_type == Mention.mention_repr_tag_name:
            mention_reprs = self.predict_reprs(data)
            mentions = list(data.mentions.values())
            for mention in mentions:
                mention_id = mention.mention_id
                mention_repr = mention_reprs[mention_id]
                data.mentions[mention_id].add_tag(output_tag, mention_repr)
        elif predict_type == Mention.mention_distance_tag_name:
            distances = self.predict_distances(data)
            data.add_mention_distances(distances, output_tag)
        else:
            raise NotImplementedError(predict_type)
        return data
