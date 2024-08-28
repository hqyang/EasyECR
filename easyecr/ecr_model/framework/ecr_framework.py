#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""



Date: 2023/11/22 16:26
"""
import os
from typing import Dict
from typing import List
from typing import Optional
import copy
import shutil
import json

from omegaconf import DictConfig

from easyecr.ecr_data.data_structure.data_structure import EcrData
from easyecr.ecr_data.data_structure.data_structure import Mention
from easyecr.ecr_model.model.ecr_model import EcrModel
from easyecr.ecr_model.cluster.cluster_model import EcrClusterModel
from easyecr.ecr_evaluate.ecr_evaluate import Evaluator
from easyecr.utils import log_utils
from easyecr.ecr_model.ecr_tagger.ecr_tagger import EcrTagger

logger = log_utils.get_logger(__file__)


class FrameworkEvalResult:
    def __init__(self, key: str, metrics: Dict):
        """

        Args:
            key:
            metrics:
        """
        self.key = key
        self.metrics = metrics

    def __repr__(self):
        result = json.dumps({"key": self.key, "metrics": self.metrics}, ensure_ascii=False)
        return result


class EcrFramework(EcrTagger):
    """
    1. 支持训练：根据配置和数据训练模型

    2. 支持预测：根据配置加载模型，对提供的数据进行预测

    3. 支持评估：根据配置加载模型，对提供的数据进行预测，提供的数据包含label，基于预测和label进行评估

    4. 训练、评估和预测时，当基础模型不直接得到聚类结果时，会依赖一个聚类模型

    关于topic
    1. train topic: 训练样本得pair只在train topic内产生
    2. predict topic: 只需要预估predict topic内pair的距离，跨topic pair间的距离为1。无论predict topic是什么，结果
    都是，每个mention都会属于某一个事件（聚类）
    3. evaluate topic: 可以单独评估每个evaluate topic内的指标，然后平均；也可以一起评估
    evaluate topic和predict topic没有任何关系

    """

    def __init__(
        self,
        predict_topic: str,
        evaluate_topic: str,
        main_metric: str,
        ecr_model: EcrModel,
        ecr_model_output_tag: str,
        cluster_model: Optional[EcrClusterModel] = None,
        evaluator: Optional[Evaluator] = None,
    ):
        """

        Args:
            conf:
            ecr_model:
            cluster_model:
            evaluator:
        """
        self.predict_topic = predict_topic
        self.evaluate_topic = evaluate_topic
        self.main_metric = main_metric
        self.ecr_model = ecr_model
        self.ecr_model_output_tag = ecr_model_output_tag
        self.cluster_model = cluster_model
        self.evaluator = evaluator

    def is_first_better_than_second(self, first: FrameworkEvalResult, second: FrameworkEvalResult):
        """

        Args:
            first:
            second:

        Returns:

        """
        metrics1 = first.metrics
        metrics2 = second.metrics
        result = metrics1[self.main_metric]["f1"] > metrics2[self.main_metric]["f1"]
        return result

    def get_the_best(self, eval_results: List[FrameworkEvalResult]) -> FrameworkEvalResult:
        """

        Args:
            eval_results:

        Returns:

        """
        best_result = None
        for eval_result in eval_results:
            if best_result is None or self.is_first_better_than_second(eval_result, best_result):
                best_result = eval_result
        return best_result

    def find_the_best_model(self, dev_data: EcrData, output_tag):
        """

        Args:
            dev_data:
            output_tag:

        Returns:

        """
        best_eval_result: FrameworkEvalResult = None
        best_version = None
        all_model_versions = self.ecr_model.get_all_versions()
        for version in all_model_versions:
            self.ecr_model.set_model_version(version)
            current_eval_results = self.evaluate(dev_data, output_tag)
            best_current_eval_result = self.get_the_best(current_eval_results)
            if best_eval_result is None or self.is_first_better_than_second(best_current_eval_result, best_eval_result):
                best_eval_result = best_current_eval_result
                best_version = version
        if best_version is not None:
            self.ecr_model.set_model_version(best_version, is_best=True)
            self.cluster_model.set_best_model(best_eval_result.key)
            logger.info(f"best_eval_result: {best_eval_result}")
        else:
            logger.info("no best version")

    def train(self, train_data: EcrData, dev_data: EcrData, output_tag: str = Mention.mention_predict_tag_name):
        """
        1. 不使用early stopping，保存每个epoch的模型，训练完后，筛选最优模型。相比于通过callback在每个epoch遍历
        聚类模型的每种超惨，这样做的优势是，基本模型和聚类模型解耦合了，在基本模型训练完后，还可以尝试不同的聚类模型的不同配置
        Args:
            train_data:
            dev_data:
            output_tag:
        Returns:

        """
        self.ecr_model.train(train_data, dev_data)
        self.find_the_best_model(dev_data, output_tag)

    def evaluate(self, data: EcrData, output_tag: str = Mention.mention_predict_tag_name) -> List[FrameworkEvalResult]:
        """给mention打上标签, 并基于ground truth进行评估

        Args:
            data:
            output_tag:

        Returns:
            key: 聚类模型超参数组成的字符串
            value: 对应配置下的评估指标
        """
        data.add_event_id()
        predicted_data = self.predict(data, output_tag)
        all_pred_tag = data.get_mention_tags_by_prefix(output_tag)
        result = []
        for tag in all_pred_tag:
            tag_metrics = self.evaluator.evaluate_from_ecr_data(predicted_data, tag, topic_tag_name=self.evaluate_topic)
            eval_result = FrameworkEvalResult(tag, tag_metrics)
            result.append(eval_result)
        return result

    def predict(self, data: EcrData, output_tag: str = Mention.mention_predict_tag_name) -> EcrData:
        """给mention打上标签
        1. 预测mention
        2. 预测共指

        :param data:
        :param output_tag:
        :return:
        """
        data_tag = self.ecr_model_output_tag
        data = self.ecr_model.predict(data, data_tag)
        data = self.cluster_model.predict(data, output_tag, data_tag, self.predict_topic)
        return data
