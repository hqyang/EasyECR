import random
import sys
from typing import List
from typing import Dict
from typing import Any
from typing import Optional
from collections import defaultdict
import copy

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

from easyecr.ecr_data.data_structure.data_structure import EcrData
from easyecr.ecr_data.data_structure.data_structure import Mention
from easyecr.utils import text_encoder
from easyecr.utils import log_utils

logger = log_utils.get_logger(__file__)


class EndToEndEventCoreferenceDataset(Dataset):

    def __init__(self, ecr_data: EcrData, coarse_type: str, train: bool = False, conf: Optional[Dict[str, Any]] = None):
        self.ecr_data = ecr_data
        self.coarse_type = coarse_type
        self.train = train
        self.conf = conf

        self.events = self.ecr_data.events
        self.mentions = self.ecr_data.mentions

        if self.train:
            self.positive_samples = []
            self.generate_positive_samples()

            # key是一个mention，value是与它不共指的mention列表
            self.negative_samples: Dict[str, List[Mention]] = defaultdict(list)
            self.include_cross_topic_negative_samples: bool = False
            self.generate_negative_samples()
            self.negative_sample_num = 4

        self.samples = []
        self.generate_samples()

        self.coarse_clusters = defaultdict(list)
        self.generate_coarse_clusters()
        self.global_clusters = {}
        self.generate_global_clusters()
        self.get_statistics()

    def get_topics(self, mention: Mention):
        """

        :param mention:
        :return:
        """
        doc_id = mention.doc_id
        doc = self.ecr_data.documents[doc_id]
        topic = doc.meta.get('doc_topic', '')
        subtopc = doc.meta.get('doc_subtopic', '')
        return topic, subtopc

    def get_statistics(self):
        """

        :return:
        """
        if self.train:
            same_subtopic = 0
            same_topic = 0
            other = 0
            for sample in self.samples:
                topic1, subtopic1 = self.get_topics(sample[0])
                topic2, subtopic2 = self.get_topics(sample[1])
                if subtopic1 == subtopic2:
                    same_subtopic += 1
                elif topic1 == topic2:
                    same_topic += 1
                else:
                    other += 1
            print(f'same_subtopic: {same_subtopic} same_topic: {same_topic} other: {other}')

    def get_all_mentions(self):
        """

        :return:
        """
        return self.mentions.values()

    def generate_coarse_clusters(self):
        """

        :return:
        """
        for event in self.events:
            topic = self.get_coarse_cluster_name(event.mentions[0])
            mentions = [self.generate_mention_key(e) for e in event.mentions]
            self.coarse_clusters[topic].append(mentions)

    def generate_global_clusters(self):
        """

        :return:
        """
        key = 'doc'
        value = []
        for event in self.events:
            mentions = [self.generate_mention_key(e) for e in event.mentions]
            value.append(mentions)
        self.global_clusters[key] = value

    def split_representations_by_coarse_cluster_label(self, representations):
        """

        :param representations:
        :return:
        """
        result = {}
        for i, cluster_label in enumerate(self.sample_coarse_cluster_label):
            if cluster_label not in result:
                result[cluster_label] = []
            result[cluster_label].append((representations[i], self.generate_mention_key(self.samples[i][0])))
        return result

    def random_chose(self, data: List):
        """

        :param data:
        :return:
        """
        if len(data) == 1:
            return data[0]
        index = random.randint(0, len(data) - 1)
        return data[index]

    def sample_negative_mention(self, mention: Mention):
        """

        Args:
            mention:

        Returns:

        """
        mention_key = self.generate_mention_key(mention)
        candidates = self.negative_samples[mention_key]
        if not candidates:
            return None
        mention = self.random_chose(candidates)
        return mention

    def get_coarse_cluster_name(self, mention: Mention):
        """

        :param mention:
        :return:
        """
        result = self.ecr_data.get_mention_tag(mention, self.coarse_type)
        return result

    def generate_positive_samples(self):
        """

        :return:
        """
        events = self.events
        for i, event in enumerate(events):
            mentions = event.mentions
            for j, mention_j in enumerate(mentions):
                for k, mention_k in enumerate(mentions):
                    if k <= j:
                        continue
                    self.positive_samples.append([mention_j, mention_k])

    def generate_mention_key(self, mention: Mention):
        """

        :param mention:
        :return:
        """
        result = mention.mention_id
        return result

    def generate_negative_samples(self):
        """

        :return:
        """
        events = self.events
        for i, event_i in enumerate(events):
            mentions_i = event_i.mentions
            for j, event_j in enumerate(events):
                if j <= i:
                    continue
                mentions_j = event_j.mentions
                for mention_i in mentions_i:
                    key_i = self.generate_mention_key(mention_i)
                    coarse_cluster_name_i = self.get_coarse_cluster_name(mention_i)
                    for mention_j in mentions_j:
                        # 只在同一个topic下生成样本
                        coarse_cluster_name_j = self.get_coarse_cluster_name(mention_j)
                        if not self.include_cross_topic_negative_samples \
                                and coarse_cluster_name_i != coarse_cluster_name_j:
                            continue
                        key_j = self.generate_mention_key(mention_j)
                        self.negative_samples[key_i].append(mention_j)
                        self.negative_samples[key_j].append(mention_i)

    def generate_samples(self):
        """
        Pair Generation for Training:
        2022-naacl-Contrastive Representation Learning for Cross-Document Coreference Resolution of Events and Entities
        2021-CL-Generalizing Cross-Document Event Coreference Resolution Across Multiple Corpora
        2020-coling-Event coreference resolution with their paraphrases and argument-aware embeddings
        2019-ACL-Revisiting Joint Modeling of Cross-document Entity and Event Coreference Resolution
        :return:
        """
        if self.train:
            unique_samples = set()
            for sample in self.positive_samples:
                sample_with_label = [sample[0], sample[1], 1]
                self.samples.append(sample_with_label)
                # sample negative pairs
                # within-document, within-subtopic, within-topic, cross-topic, within-cluster
                for _ in range(self.negative_sample_num):
                    negative_mention = self.sample_negative_mention(sample_with_label[0])
                    if negative_mention:
                        key1 = self.generate_mention_key(sample_with_label[0])
                        key2 = self.generate_mention_key(negative_mention)
                        if (key1, key2) in unique_samples or (key2, key1) in unique_samples:
                            continue
                        else:
                            unique_samples.add((key1, key2))
                            unique_samples.add((key2, key1))
                        self.samples.append([sample_with_label[0], negative_mention, 0])
        else:
            mentions = list(self.mentions.values())
            for i in range(len(mentions)):
                for j in range(i + 1, len(mentions)):
                    mention_i = mentions[i]
                    mention_j = mentions[j]

                    coarse_cluster_name_i = self.get_coarse_cluster_name(mention_i)
                    coarse_cluster_name_j = self.get_coarse_cluster_name(mention_j)
                    if coarse_cluster_name_i != coarse_cluster_name_j:
                        continue
                    self.samples.append([mention_i, mention_j, 0])

    def __len__(self):
        result = len(self.samples)
        return result

    def get_mention_repr(self, mention: Mention):
        """

        Args:
            mention:

        Returns:

        """
        meta = mention.meta
        mention_hiddens = meta['mention_hiddens']
        hiddens_mask = [1] * mention_hiddens.shape[1]
        mention_length = meta['mention_length']
        mention_hiddens_first = meta['mention_hiddens_first']
        mention_hiddens_last = meta['mention_hiddens_last']
        result = [
            np.squeeze(mention_hiddens, axis=0),
            hiddens_mask,
            [mention_length],
            np.squeeze(mention_hiddens_first, axis=0),
            np.squeeze(mention_hiddens_last, axis=0)
        ]
        return result

    def __getitem__(self, idx: int):
        """

        :param idx:
        :return:
        """
        sample = self.samples[idx]
        result = []
        first_tokenized_result = self.get_mention_repr(sample[0])
        result.extend(first_tokenized_result)
        result.extend(self.get_mention_repr(sample[1]))
        result.append([sample[2]])
        result = [torch.tensor(e) for e in result]
        return result

    @staticmethod
    def collate_fn(data):
        """

        :param data:
        :return:
        """
        field_num = len(data[0])
        fields = []
        for i in range(field_num):
            temp = [e[i] for e in data]
            if temp[0] is not None:
                field_data = pad_sequence(temp, padding_value=0, batch_first=True)
            else:
                field_data = None
            fields.append(field_data)
        return fields
