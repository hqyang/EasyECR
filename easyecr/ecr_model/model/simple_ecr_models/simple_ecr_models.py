#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""



Date: 2023/12/14 17:35
"""

import os
from typing import Dict
from typing import List
from typing import Tuple
from typing import Any
import math
import shutil
from enum import Enum
from multiprocessing import Pool

from omegaconf import DictConfig
from tqdm import tqdm
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from easyecr.ecr_data.data_structure.data_structure import EcrData
from easyecr.ecr_data.data_structure.data_structure import Mention
from easyecr.ecr_model.model.ecr_model import EcrModel
from easyecr.utils import log_utils
from easyecr.ecr_model.sample_generator.sample_generator import SimpleGenerator

logger = log_utils.get_logger(__file__)


class SimpleEcrModel(EcrModel):
    """不用基于上层的ecr指标选择最优版本的模型"""

    def get_all_versions(self) -> List[str]:
        """获得模型训练过程中保存的所有模型版本
        配合set_model_version，使得外层框架可以确定最优模型版本

        Returns: version用str表示
        """
        result = []
        return result

    def set_model_version(self, version: int, is_best: bool = False):
        """设定当前需要使用的模型版本
        配合get_all_versions，使得外层框架可以确定最优模型版本
        Args:
            version:
            is_best:

        Returns:

        """
        pass


class SimpleDistanceEcrModel(SimpleEcrModel):
    """预测结果为mention pair间的距离"""

    def __init__(self, train_topic: str, predict_topic: str, thread_num: int = 10):
        """

        Args:
            train_topic:
            predict_topic:
            thread_num:
        """
        self.train_topic = train_topic
        self.predict_topic = predict_topic
        self.thread_num = thread_num

    def predict_mention_distance(self, mention1: Mention, mention2: Mention, data: EcrData) -> float:
        """

        Args:
            mention1:
            mention2:
            data:

        Returns:

        """
        raise NotImplementedError()

    def predict_distances_single_process(self, data: EcrData) -> Dict[str, Dict[str, float]]:
        """


        Args:
            data:

        Returns:

        """
        sample_generator = SimpleGenerator(times=0)
        samples = sample_generator.generate(data, topic_name=self.predict_topic)
        samples = samples["positive"] + samples["negative"]
        logger.info(f"'predict samples num: {len(samples)}")
        distances = {}
        flatten_distances = []
        for mention1, mention2 in tqdm(samples):
            id1 = mention1.mention_id
            id2 = mention2.mention_id
            if id1 not in distances:
                distances[id1] = {}
            if id2 not in distances:
                distances[id2] = {}
            distance = self.predict_mention_distance(mention1, mention2, data)
            flatten_distances.append(distance)
            distances[id1][id2] = distance
            distances[id2][id1] = distance
        logger.info(f"'predict positive distance num: {len(flatten_distances) - sum(flatten_distances)}")
        return distances

    @staticmethod
    def split(elements: List[Any], part_num: int):
        """

        Args:
            elements:
            part_num:

        Returns:

        """
        result = []
        split_size = math.ceil(len(elements) / part_num)
        start = 0
        while start < len(elements):
            end = start + split_size
            result.append(elements[start:end])
            start = end
        return result

    @staticmethod
    def predict_distances_for_samples(parametes) -> Dict[str, Dict[str, float]]:
        """

        Args:
            parametes:

        Returns:

        """
        ecr_model, data, samples = parametes
        distances = {}
        flatten_distances = []
        for mention1, mention2 in tqdm(samples):
            id1 = mention1.mention_id
            id2 = mention2.mention_id
            if id1 not in distances:
                distances[id1] = {}
            if id2 not in distances:
                distances[id2] = {}
            distance = ecr_model.predict_mention_distance(mention1, mention2, data)
            flatten_distances.append(distance)
            distances[id1][id2] = distance
            distances[id2][id1] = distance
        logger.info(f"'predict positive distance num: {len(flatten_distances) - sum(flatten_distances)}")
        return distances

    def predict_distances_parallel(self, data: EcrData) -> Dict[str, Dict[str, float]]:
        """


        Args:
            data:

        Returns:

        """
        sample_generator = SimpleGenerator(times=0)
        samples = sample_generator.generate(data, topic_name=self.predict_topic)
        samples = samples["positive"] + samples["negative"]
        logger.info(f"'predict samples num: {len(samples)}")
        groups = SimpleDistanceEcrModel.split(samples, self.thread_num)
        group_parameters = tuple([(self, data, group) for group in groups])
        with Pool(self.thread_num) as p:
            distances_parts = p.map(SimpleDistanceEcrModel.predict_distances_for_samples, group_parameters)

        distances = {}
        for distances_part in distances_parts:
            for mention_id, distances_part_part in distances_part.items():
                if mention_id not in distances:
                    distances[mention_id] = {}
                distances[mention_id].update(distances_part_part)
        return distances

    def predict_distances(self, data: EcrData) -> Dict[str, Dict[str, float]]:
        """


        Args:
            data:

        Returns:

        """
        # distances = self.predict_distances_single_process(data)
        distances = self.predict_distances_parallel(data)
        return distances

    def get_predict_type(self) -> str:
        """

        Returns: repr, distance

        """
        result = Mention.mention_distance_tag_name
        return result


class SimpleReprEcrModel(SimpleEcrModel):
    """预测结果为mention的表示"""

    def predict_mention_repr(self, mention: Mention, data: EcrData) -> List[float]:
        """

        Args:
            mention:
            data:

        Returns:

        """
        raise NotImplementedError()

    def predict_reprs(self, data: EcrData) -> Dict[str, List[float]]:
        """


        Args:
            data:
            output_tag:

        Returns:

        """
        mentions = list(data.mentions.values())
        result = {}
        for mention in mentions:
            mention_id = mention.mention_id
            mention_repr = self.predict_mention_repr(mention, data)
            result[mention_id] = mention_repr
        return result

    def get_predict_type(self) -> str:
        """

        Returns: repr, distance

        """
        result = Mention.mention_repr_tag_name
        return result
