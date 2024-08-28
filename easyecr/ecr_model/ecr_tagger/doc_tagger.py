#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""



Date: 2023/12/16 9:31
"""
from easyecr.ecr_model.ecr_tagger.ecr_tagger import EcrTagger
from easyecr.ecr_data.data_structure.data_structure import EcrData
from easyecr.pipeline import load_lemma_data


class DocTagger(EcrTagger):
    def train(self, train_data: EcrData, dev_data: EcrData):
        """

        Args:
            train_data:
            dev_data:

        Returns:

        """
        pass

    def predict(self, data: EcrData, output_tag: str) -> EcrData:
        """

        Args:
            data:
            output_tag:
            **kwargs:

        Returns:

        """
        evt_mention_map = load_lemma_data.load_mention_map()
        mentions = list(data.mentions.values())
        for mention in mentions:
            key = load_lemma_data.get_mention_id(mention, data)
            predicted_topic = evt_mention_map[key]["predicted_topic"]
            mention.add_tag(output_tag, predicted_topic)
        return data
