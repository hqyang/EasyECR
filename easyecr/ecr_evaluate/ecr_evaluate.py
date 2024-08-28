#!/usr/bin/env python
# -*- coding:utf-8 -*-　
"""

https://huggingface.co/spaces/evaluate-metric/coval
https://github.com/ns-moosavi/coval




Date: 2023/10/30 16:39
"""
from typing import List
from typing import Dict

import numpy as np

from external_code.coval import scorer
from external_code.coval.coval.eval import evaluator
from easyecr.ecr_data.data_structure.data_structure import EcrData
from easyecr.ecr_data.data_structure.data_structure import Mention


class Evaluator:
    """
    1. 支持直接评估，分topic
    2. 支持直接评估，不分topic
    3. 支持基于topic的平均
    """
    def __init__(self, average_over_topic: bool = False,
                 metric_names: List[str] = ['mentions', 'muc', 'bcub', 'ceafe', 'lea'],
                 keep_singletons: bool = False):
        """

        Args:
            average_over_topic: 如果把topic_tag看作数据集名，average_over_topic为True类似于Macro平均，
            metric_names:
            keep_singletons:
        """
        self.average_over_topic = average_over_topic
        name_function_mappings = {
            'mentions': evaluator.mentions,
            'muc': evaluator.muc,
            'bcub': evaluator.b_cubed,
            'ceafe': evaluator.ceafe,
            'lea': evaluator.lea
        }
        self.metrics = [(name, name_function_mappings[name]) for name in metric_names]
        self.keep_singletons = keep_singletons

    def average_topic_metrics(self, all_topic_metrics: List[Dict[str, Dict]]):
        """

        :param all_topic_metrics:
        :return:
        """
        num = len(all_topic_metrics)
        result = all_topic_metrics[0]
        for topic_metrics in all_topic_metrics[1:]:
            for metric_name, metrics in topic_metrics.items():
                for sub_metric_name, sub_metric in metrics.items():
                    result[metric_name][sub_metric_name] += sub_metric

        for metric_name, metrics in result.items():
            for sub_metric_name, sub_metric in metrics.items():
                result[metric_name][sub_metric_name] = sub_metric / num
        return result

    def evaluate(self, gold: Dict[str, List[str]], pred: Dict[str, List[str]]):
        """

        :param gold:
        :param pred:
        :return:
        """
        if self.average_over_topic:
            all_topic_metrics = []
            for topic, cluster_result in gold.items():
                topic_gold = {topic: cluster_result}
                topic_pred = {topic: pred[topic]}
                topic_metrics = scorer.evaluate_from_clusters(topic_pred, topic_gold, self.metrics,
                                                              self.keep_singletons)
                all_topic_metrics.append(topic_metrics)
            result = self.average_topic_metrics(all_topic_metrics)
        else:
            result = scorer.evaluate_from_clusters(pred, gold, self.metrics, self.keep_singletons)
        return result

    def evaluate_from_mention_representation(self, gold: Dict[str, List[str]],
                                             pred: Dict[str, List[np.ndarray]],
                                             mention_keys: Dict[str, List[str]],
                                             cluster_model):
        """

        :param gold:
        :param pred:
        :param mention_keys:
        :param cluster_model:
        :return:
        """
        pass

    def evaluate_from_mention_pair_distance(self, gold: Dict[str, List[str]],
                                            pred: Dict[str, np.ndarray],
                                            mention_keys: Dict[str, List[str]],
                                            cluster_model):
        """

        :param gold:
        :param pred:
        :param mention_keys:
        :param cluster_model:
        :return:
        """
        pass

    def get_mention_clusters(self, ecr_data: EcrData, tag: str, topic_tag_name: str):
        """

        Args:
            ecr_data:
            tag:
            topic_tag_name:

        Returns:

        """
        topic_and_mentions = ecr_data.group_mention_by_tag(topic_tag_name)
        result = {}
        for topic, mentions in topic_and_mentions.items():
            clusters = {}
            for mention in mentions:
                cluster_id = ecr_data.get_mention_tag(mention, tag)
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(mention.mention_id)
            clusters = list(clusters.values())
            result[topic] = clusters
        return result

    def evaluate_from_ecr_data(self, ecr_data: EcrData, pred_tag: str, gold_tag: str = Mention.mention_label_tag_name,
                               topic_tag_name: str = Mention.basic_topic_tag_name):
        """

        Args:
            ecr_data:
            pred_tag:
            gold_tag:
            topic_tag_name: topic_tag可以看作数据集名

        Returns:

        """
        gold_clusters = self.get_mention_clusters(ecr_data, gold_tag, topic_tag_name)
        pred_clusters = self.get_mention_clusters(ecr_data, pred_tag, topic_tag_name)
        result = self.evaluate(gold_clusters, pred_clusters)
        return result


def evaluate(gold: Dict[str, List[str]], pred: Dict[str, List[str]], keep_singletons: bool = False):
    """

    :param gold:
    :param pred:
    :return:
    """
    # gold = {key: [e for e in value if len(e) > 1] for key, value in gold.items()}
    # pred = {key: [e for e in value if len(e) > 1] for key, value in pred.items()}
    all_metrics = [('mentions', evaluator.mentions), ('muc', evaluator.muc),
                   ('bcub', evaluator.b_cubed), ('ceafe', evaluator.ceafe),
                   ('lea', evaluator.lea)]
    result = scorer.evaluate_from_clusters(pred, gold, all_metrics, keep_singletons)
    return result


if __name__ == '__main__':
    gold = [['News'], ['Emory University'], ['confirmed'],
            ['yesterday'], ['announcement'], ['name', 'approached'],
            ['names', 'nominates', 'decision']]
    p1 = [['News'], ['Emory University'], ['confirmed'],
            ['yesterday'], ['announcement', 'name', 'approached',
            'names', 'nominates', 'decision']]
    p2 = [['News that'], ['Emory'],
            ['announcement', 'name', 'approached'],
            ['names', 'nominates', 'decision']]
    print(evaluate({'doc': gold}, {'doc': p1}))
    print(evaluate({'doc': gold}, {'doc': p2}))