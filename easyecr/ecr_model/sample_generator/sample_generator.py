#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""



Date: 2023/12/13 17:37
"""
from typing import Any
from typing import Optional
from typing import List
from typing import Dict
from typing import Tuple

import numpy as np
from numpy import random

from easyecr.ecr_data.data_structure.data_structure import EcrData
from easyecr.ecr_data.data_structure.data_structure import Mention


class SampleGenerator:
    def generate(
        self,
        data: EcrData,
        topic_name: str,
        include_singleton: bool = True,
        label_tag_name: str = Mention.mention_label_tag_name,
        distance_tag_name: Optional[str] = None,
    ) -> Dict[str, List[Tuple[Mention, Mention]]]:
        """

        Args:
            data:
            topic_name: 所有pair都限定在相同的topic内
            include_singleton: 是否包含单例mention。之前的部分方法，在评估时，评估集合直接去掉单例，这不合理，这个选项支持对这种设定的复现
            label_tag_name:
            distance_tag_name: 当需要基于其它信息来生成样本时，比如mention间相似度，mention表示，这个字段指定具体要用的信息类型

        Returns:

        """
        raise NotImplementedError()


class SimpleGenerator(SampleGenerator):
    """
    正样本+采样负样本
    1. 所有正样本
    2. n * 正样本数 负样本

    times为0或None时，不采样

    也可生成预测样本：times为0或None, label_tag_name为''或None
    """

    def __init__(self, times: Optional[int]):
        """

        Args:
            times: 负样本数相对正样本数的倍数，None或0表示不采样
        """
        self.times = times

    def generate(
        self,
        data: EcrData,
        topic_name: str,
        include_singleton: bool = True,
        label_tag_name: str = Mention.mention_label_tag_name,
        distance_tag_name: Optional[str] = None,
    ) -> Dict[str, List[Tuple[Mention, Mention]]]:
        """

        Args:
            data:
            topic_name:
            include_singleton:
            label_tag_name:
            distance_tag_name:

        Returns:

        """
        # mention_pairs = data.get_mention_pairs(topic_name, include_singleton)
        mention_pairs = data.get_mention_pairs_parallel(topic_name, include_singleton, thread_num=20)
        positive_samples = []
        negative_samples = []
        for mention1, mention2 in mention_pairs:
            if label_tag_name:
                event_id1 = data.get_mention_tag(mention1, label_tag_name)
                event_id2 = data.get_mention_tag(mention2, label_tag_name)
                label = int(str(event_id1) == str(event_id2))
                if label:
                    positive_samples.append([mention1, mention2])
                else:
                    negative_samples.append([mention1, mention2])
            else:
                negative_samples.append([mention1, mention2])
        if self.times:
            negative_sample_num = int(len(positive_samples) * self.times)
            negative_sample_ints = [i for i in range(len(negative_samples))]
            sampled_negative_sample_ints = random.choice(negative_sample_ints, negative_sample_num, replace=True)
            sampled_negative_samples = [negative_samples[i] for i in sampled_negative_sample_ints]
        else:
            sampled_negative_samples = negative_samples
        result = {"positive": positive_samples, "negative": sampled_negative_samples}
        print(f"positive sample num: {len(positive_samples)} negative sample num: {len(negative_samples)}")
        return result


class HardGenerator(SampleGenerator):
    """
    2022-NAACL-Contrastive Representation Learning for Cross-Document Coreference Resolution of Events and Entities
    1. 针对每个mention，选取离它最远的共指mention形成pair，构成正样本；选取离它最近的非共指mention形成pair，构成负样本
    2. 对负样本下采样，只留下相似度大于所有正样本相似度中位数的负样本
    """

    def __init__(self, negative_sample_num_per_positive_sample: int = 4):
        """

        Args:
            negative_sample_num_per_positive_sample:
        """
        self.negative_sample_num_per_positive_sample = negative_sample_num_per_positive_sample

    def generate(
        self,
        data: EcrData,
        topic_name: str,
        include_singleton: bool = True,
        label_tag_name: str = Mention.mention_label_tag_name,
        distance_tag_name: Optional[str] = None,
    ) -> Dict[str, List[Tuple[Mention, Mention]]]:
        """

        Args:
            data:
            topic_name:
            include_singleton:
            label_tag_name:
            distance_tag_name:

        Returns:

        """
        mention_pairs = data.get_mention_pairs_parallel(topic_name, include_singleton)
        positive_samples = {}
        positive_distances = []
        negative_samples = {}
        for mention1, mention2 in mention_pairs:
            event_id1 = data.get_mention_tag(mention1, label_tag_name)
            event_id2 = data.get_mention_tag(mention2, label_tag_name)
            label = int(event_id1 == event_id2)
            if label:
                if mention1.mention_id not in positive_samples:
                    positive_samples[mention1.mention_id] = []
                if mention2.mention_id not in positive_samples:
                    positive_samples[mention2.mention_id] = []
                positive_samples[mention1.mention_id].append([mention1, mention2])
                positive_samples[mention2.mention_id].append([mention2, mention1])

                distance = mention1.get_tag(distance_tag_name)[mention2.mention_id]
                positive_distances.append(distance)
            else:
                if mention1.mention_id not in negative_samples:
                    negative_samples[mention1.mention_id] = []
                if mention2.mention_id not in negative_samples:
                    negative_samples[mention2.mention_id] = []
                negative_samples[mention1.mention_id].append([mention1, mention2])
                negative_samples[mention2.mention_id].append([mention2, mention1])

        print(f"positive sample num: {len(positive_samples)} negative sample num: {len(negative_samples)}")
        sampled_positive_samples = []
        for mention_id, pairs in positive_samples.items():
            distances = []
            for mention1, mention2 in pairs:
                distance = mention1.get_tag(distance_tag_name)[mention2.mention_id]
                distances.append(distance)
            target_index = np.argmax(distances)
            sampled_positive_samples.append(pairs[target_index])

        sampled_negative_samples = []
        sampled_negative_sample_id_pairs = set()
        for mention_id, pairs in negative_samples.items():
            distances = []
            for mention1, mention2 in pairs:
                distance = mention1.get_tag(distance_tag_name)[mention2.mention_id]
                distances.append(distance)
            sorted_indices = np.argsort(distances)
            for target_index in sorted_indices[: self.negative_sample_num_per_positive_sample]:
                mention1, mention2 = pairs[target_index]
                id_pair = (mention1.mention_id, mention2.mention_id)
                if id_pair in sampled_negative_sample_id_pairs:
                    continue
                else:
                    sampled_negative_samples.append(pairs[target_index])
                    sampled_negative_sample_id_pairs.add(id_pair)

        positive_distance_median = np.median(positive_distances)
        final_sampled_negative_samples = []
        for mention1, mention2 in sampled_negative_samples:
            distance = mention1.get_tag(distance_tag_name)[mention2.mention_id]
            if distance < positive_distance_median:
                final_sampled_negative_samples.append([mention1, mention2])

        result = {"positive": sampled_positive_samples, "negative": final_sampled_negative_samples}
        return result


class NearestNeighborGenerator(SampleGenerator):
    """
    2023-findings-of-ACL-2∗n is better than n2: Decomposing Event Coreference Resolution into Two Tractable Problems
    方法：
    1. 生成所有mention pair
    2. 通过简单的相似度计算，只保留相似度大于一定阈值的pair，有正样本和负样本

    2021-EMNLP-Focus on what matters: Applying Discourse Coherence Theory to Cross Document Coreference
    方法：
    针对每个mention，选取k近邻构成pair，有正样本和负样本

    """

    def __init__(self, threshold: Optional[float] = None, top_k: Optional[int] = None):
        """

        Args:
            threshold: threshold和top_k必须只有一个不为空
            top_k: threshold和top_k必须只有一个不为空
        """
        assert (threshold is not None and top_k is None) or (threshold is None and top_k is not None)
        self.threshold = threshold
        self.top_k = top_k

    def generate(
        self,
        data: EcrData,
        topic_name: str,
        include_singleton: bool = True,
        label_tag_name: str = Mention.mention_label_tag_name,
        distance_tag_name: Optional[str] = None,
    ) -> Dict[str, List[Tuple[Mention, Mention]]]:
        """

        Args:
            data:
            topic_name:
            include_singleton:
            label_tag_name:
            distance_tag_name:

        Returns:

        """
        if data.name == "wec":
            print("get_mention_pairs_parallel start!")
            mention_pairs = data.get_mention_pairs_parallel(topic_name, include_singleton, thread_num=16)
            print("get_mention_pairs_parallel finish!")
        else:
            mention_pairs = data.get_mention_pairs(topic_name, include_singleton)
        if self.threshold is not None:
            positive_samples = []
            negative_samples = []
            for mention1, mention2 in mention_pairs:
                try:
                    distance = mention1.get_tag(distance_tag_name)[mention2.mention_id]
                    if distance < self.threshold:
                        event_id1 = data.get_mention_tag(mention1, label_tag_name)
                        event_id2 = data.get_mention_tag(mention2, label_tag_name)
                        label = int(event_id1 == event_id2)
                        if label:
                            positive_samples.append([mention1, mention2])
                        else:
                            negative_samples.append([mention1, mention2])
                    else:
                        pass
                except KeyError:
                    continue
        else:
            samples = {}
            for mention1, mention2 in mention_pairs:
                if mention1.mention_id not in samples:
                    samples[mention1.mention_id] = []
                if mention2.mention_id not in samples:
                    samples[mention2.mention_id] = []
                samples[mention1.mention_id].append([mention1, mention2])
                samples[mention2.mention_id].append([mention2, mention1])

            positive_samples = []
            negative_samples = []
            for mention_id, pairs in samples.items():
                distances = []
                for mention1, mention2 in pairs:
                    distance = mention1.get_tag(distance_tag_name)[mention2.mention_id]
                    distances.append(distance)
                sorted_indices = np.argsort(distances)
                target_pairs = [pairs[index] for index in sorted_indices[: self.top_k]]
                for mention1, mention2 in target_pairs:
                    event_id1 = data.get_mention_tag(mention1, label_tag_name)
                    event_id2 = data.get_mention_tag(mention2, label_tag_name)
                    label = int(event_id1 == event_id2)
                    if label:
                        positive_samples.append([mention1, mention2])
                    else:
                        negative_samples.append([mention1, mention2])

        result = {"positive": positive_samples, "negative": negative_samples}

        return result
