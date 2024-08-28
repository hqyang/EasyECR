#!/usr/bin/env python
# -*- coding:utf-8 -*-　
"""



Date: 2023/12/16 23:44
"""
from typing import List
from typing import Dict

from easyecr.ecr_model.model.ecr_model import EcrModel
from easyecr.ecr_data.data_structure.data_structure import Mention
from easyecr.ecr_data.data_structure.data_structure import EcrData
from easyecr.utils import log_utils

logger = log_utils.get_logger(__file__)


class DistanceEcrModel(EcrModel):

    def predict_distance(self, data: EcrData) -> Dict[str, Dict[str, float]]:
        """

        Args:
            data:

        Returns:

        """
        raise NotImplementedError()

    def predict(self, data: EcrData, output_tag: str) -> EcrData:
        """

        Args:
            data:
            output_tag:

        Returns:

        """
        distances = self.predict_distance(data)
        data.add_mention_distances(distances, output_tag)
        return data

    def get_predict_type(self) -> str:
        """

        Returns: repr, distance

        """
        result = Mention.mention_distance_tag_name
        return result


class ReprEcrModel(EcrModel):
    def predict_repr(self, data: EcrData) -> Dict[str, List[float]]:
        """

        Args:
            data:

        Returns:

        """
        raise NotImplementedError()

    def predict(self, data: EcrData, output_tag: str) -> EcrData:
        """

        Args:
            data:
            output_tag:

        Returns:

        """
        mention_id_repr = self.predict_repr(data)
        mentions = list(data.mentions.values())
        for mention in mentions:
            mention_id = mention.mention_id
            mention_repr = mention_id_repr[mention_id]
            data.mentions[mention_id].add_tag(output_tag, mention_repr)
        return data

    def get_predict_type(self) -> str:
        """

        Returns: repr, distance

        """
        result = Mention.mention_repr_tag_name
        return result
