#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""



Date: 2023/12/14 17:41
"""
from typing import List
from typing import Tuple
from typing import Set

from sklearn.metrics.pairwise import cosine_distances

from easyecr.ecr_model.model.simple_ecr_models.simple_ecr_models import SimpleDistanceEcrModel
from easyecr.ecr_data.data_structure.data_structure import EcrData
from easyecr.ecr_data.data_structure.data_structure import Mention
from easyecr.ecr_model.sample_generator.sample_generator import SimpleGenerator
from easyecr.utils import log_utils
from easyecr.pipeline import load_lemma_data

logger = log_utils.get_logger(__file__)


class EmbeddingDistance(SimpleDistanceEcrModel):
    def __init__(self, train_topic: str, predict_topic: str, repr_tag: str):
        """

        Args:
            train_topic:
            predict_topic:
            repr_tag:
        """
        super().__init__(train_topic, predict_topic)
        self.repr_tag = repr_tag

    def train(self, train_data: EcrData, dev_data: EcrData):
        """

        Args:
            train_data:
            dev_data:

        Returns:

        """
        pass

    def predict_mention_distance(self, mention1: Mention, mention2: Mention, data: EcrData) -> float:
        """

        Args:
            mention1:
            mention2:
            data:

        Returns:

        """
        embedding1 = mention1.get_tag(self.repr_tag)
        embedding2 = mention2.get_tag(self.repr_tag)
        result = cosine_distances([embedding1], [embedding2])[0][0]
        return result
