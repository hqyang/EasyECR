#!/usr/bin/env python
# -*- coding:utf-8 -*-　
"""



Date: 2023/12/13 10:09
"""
import random
import sys
from collections import defaultdict
import copy
from typing import Dict
from typing import List

from omegaconf import DictConfig
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoConfig
from transformers import AutoModel
from transformers import get_linear_schedule_with_warmup
import torch
from torch import nn
import pytorch_lightning as pl

from easyecr.ecr_data.data_structure.data_structure import EcrData
from easyecr.ecr_data.data_structure.data_structure import Mention
from easyecr.ecr_model.model.event_coreference_pl_module import EventCoreferenceModule
from easyecr.ecr_model.model.pl_ecr_models.pl_ecr_model import PlEcrModel
from easyecr.ecr_data.data_structure.data_structure import EcrData
from easyecr.ecr_data.data_structure.data_structure import Mention
from easyecr.utils import text_encoder
from easyecr.utils import log_utils

logger = log_utils.get_logger(__file__)


class EventCoreferenceDataset(Dataset):

    def __init__(self, ecr_data: EcrData, tokenizer, coarse_type: str, train: bool = False,
                 embedding_model_name: str = 'sgpt', embedding_model_size: str = '125m', device: int = 0,
                 without_singleton: bool = False):
        self.ecr_data = ecr_data
        self.tokenizer: AutoTokenizer = tokenizer
        self.coarse_type = coarse_type
        self.train = train
        self.without_singleton = without_singleton
        self.events = self.ecr_data.events
        self.filter_events()

        self.mentions = {}
        self.generate_all_mentions()

        self.mention_and_text = {}
        self.generate_mention_and_text()

        if self.train:
            self.mention_and_embedding = {}
            self.text_encoder = None
            if embedding_model_name:
                self.text_encoder = text_encoder.TextEncoder(model_name=embedding_model_name,
                                                             version=embedding_model_size,
                                                             device=device)
                self.embedding_batch_size = 32
                self.max_text_len = 512
            if self.text_encoder is not None:
                self.generate_mention_and_embedding()

            self.positive_samples = []
            self.positive_similarities = []
            self.generate_positive_samples()
            logger.info(f'filter positive samples: {len(self.positive_similarities)}->{len(self.positive_samples)}')
            self.positive_similarity_median = np.median(self.positive_similarities)

            # key是一个mention，value是与它不共指的mention列表
            self.negative_samples: Dict[str, List[Mention]] = defaultdict(list)
            self.include_cross_topic_negative_samples: bool = False
            self.generate_negative_samples()
            self.filter_negative_samples()
            self.negative_sample_num = 4
        self.samples = []
        self.samples_clean = []
        self.sample_coarse_cluster_label = []
        self.generate_samples()
        self.coarse_clusters = defaultdict(list)
        self.generate_coarse_clusters()
        self.global_clusters = {}
        self.generate_global_clusters()
        self.get_statistics()

    def filter_events(self):
        """

        :return:
        """
        if self.without_singleton:
            temp = []
            for event in self.events:
                if len(event.mentions) < 2:
                    continue
                temp.append(event)
            self.events = temp

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

    def generate_all_mentions(self):
        """

        :return:
        """
        events = self.events
        for event in events:
            mentions = event.mentions
            for mention in mentions:
                key = self.generate_mention_key(mention)
                self.mentions[key] = mention

    def get_all_mentions(self):
        """

        :return:
        """
        return self.mentions.values()

    def generate_mention_and_text(self):
        """

        :return:
        """
        mentions = self.get_all_mentions()
        for mention in mentions:
            mention_key = self.generate_mention_key(mention)
            text = self.generate_mention_text(mention)
            self.mention_and_text[mention_key] = text

    def generate_mention_and_embedding(self):
        """

        :return:
        """
        keys = []
        texts = []
        for key, text in self.mention_and_text.items():
            keys.append(key)
            texts.append(' '.join(text))

        embeddings_list = []
        start = 0
        while start < len(texts):
            end = start + self.embedding_batch_size
            temp = self.text_encoder.encode(texts[start: end], max_text_len=self.max_text_len)
            embeddings_list.append(temp)
            start = end

        embeddings = torch.cat(embeddings_list, dim=0).numpy()
        for i in range(len(keys)):
            self.mention_and_embedding[keys[i]] = embeddings[i]

    def find_target_partner(self, anchor: str, candiates: List[str], pair_type):
        """

        :param anchor:
        :param candiates:
        :param pair_type:
        :return:
        """
        pass
        # anchor_text =

        # embeddings = self.text_encoder.encode([explanation] + meanings, max_text_len=128).numpy()
        # meaning_embeddings = embeddings[1:]
        # explanation_embedding = embeddings[0]
        # similarities = [self.text_encoder.compute_cosine_similarity([explanation_embedding, meaning_embeddings[i]])
        #                 for i in range(len(glosses))]
        # result_index = np.argmax(similarities)
        # result = keys[result_index]
        # result = [result]
        # return result
        pass

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

        :param current_event_index:
        :return:
        """
        mention_key = self.generate_mention_key(mention)
        candidates = self.negative_samples[mention_key]
        if not candidates:
            return None
        mention = self.random_chose(candidates)
        return mention

    def add_special_chars(self, s, start_end_labels, s_type: str = 'char'):
        """

        :param s:
        :param start_end_labels:
        :param s_type:
        :return:
        """
        start_end_labels = sorted(start_end_labels, key=lambda x: x[0], reverse=True)
        for start, end, label in start_end_labels:  # 使用reversed是因为从前向后修改将会偏移后面的索引
            if s_type == 'word':
                result = []
                result.append(' '.join(s[:start] + [f'<{label}>']))
                result.append(' '.join(s[start:end]))
                result.append(' '.join([f'</{label}>'] + s[end:]))
                # s = s[:start] + [f'<{label}>'] + s[start:end] + [f'</{label}>'] + s[end:]
            else:
                result = [s[:start] + f'<{label}>', s[start:end], f'</{label}>' + s[end:]]
                # s = s[:start] + f'<{label}>' + s[start:end] + f'</{label}>' + s[end:]
        return result

    def convert_extent_to_tuple(self, extent: dict, mention_start: int):
        """

        :param extent:
        :param mention_start:
        :return:
        """
        extent_text, extent_start, extent_end = extent.text, extent.start, extent.end
        extent_start = int(extent_start)
        extent_end = int(extent_end)
        start_end = [extent_start - mention_start, extent_end - mention_start + 1]
        return start_end

    def generate_annotation_template_str(self, annotation_template: dict):
        """

        :param annotation_template:
        :return:
        """
        annotation_template = copy.deepcopy(annotation_template)
        for key, value in annotation_template.items():
            if value:
                if key in ['时间状语', '地点状语']:
                    value = f'{value}, '
                else:
                    value = f'{value} '
                annotation_template[key] = value
        result = '{时间状语}{地点状语}{涉事主体}{谓语}{宾语}'.format_map(annotation_template)
        return result

    def generate_mention_text(self, mention):
        """

        :param mention:
        :return:
        """
        if self.ecr_data.name == 'ECBPlus' or self.ecr_data.meta['index_type'] == 'word':
            mention_start = int(mention.extent.start)
            words = mention.extent.words
            start_end_labels = []
            tigger_tuple = self.convert_extent_to_tuple(mention.anchor, mention_start) + ['trigger']
            start_end_labels.append(tigger_tuple)
            # if mention.arguments:
            #     for argument in mention.arguments:
            #         role = argument['role']
            #         a_text = argument['text']
            #         a_tuple = self.convert_extent_to_tuple(a_text, mention_start) + [role]
            #         start_end_labels.append(a_tuple)
            changed_words = self.add_special_chars(words, start_end_labels, s_type='word')
            changed_text = changed_words
        else:
            mention_start = int(mention.extent.start)
            mention_text = mention.extent.text
            start_end_labels = []
            tigger_tuple = self.convert_extent_to_tuple(mention.anchor, mention_start) + ['trigger']
            start_end_labels.append(tigger_tuple)
            # if mention.arguments:
            #     for argument in mention.arguments:
            #         role = argument['role']
            #         a_text = argument['text']
            #         a_tuple = self.convert_extent_to_tuple(a_text, mention_start) + [role]
            #         start_end_labels.append(a_tuple)
            changed_text = self.add_special_chars(mention_text, start_end_labels)
        if self.coarse_type != 'doc':
            if self.ecr_data.name == 'ideacar':
                # arguments = mention.arguments
                # context = self.generate_annotation_template_str(arguments)
                if len(''.join(changed_text)) < 100:
                    context = self.ecr_data.documents[mention.doc_id].text[:100]
                    changed_text[0] = f'{context}。{changed_text[0]}'
            else:
                context_words = [e['text'] for e in self.ecr_data.documents[mention.doc_id].meta['doc_token']
                                 if e['sentence'] in ['0', '1']]
                context = ' '.join(context_words)
                changed_text[0] = f'{context} {changed_text[0]}'
        result = changed_text
        if len(''.join(result)) > 500:
            result[0] = result[0][-200:]
            result[-1] = result[-1][:200]
        return result

    def converted_to_texts(self, clusters: List[List[str]]):
        """

        :param clusters:
        :return:
        """
        result = []
        for cluster in clusters:
            converted_cluster = [' '.join(self.mention_and_text[e]) for e in cluster]
            result.append(converted_cluster)
        return result

    def get_coarse_cluster_name(self, mention: Mention):
        """

        :param mention:
        :return:
        """
        if self.coarse_type == 'doc':
            result = mention.doc_id
        elif not self.coarse_type or self.coarse_type == 'all':
            result = 'all'
        else:
            document = self.ecr_data.documents[mention.doc_id]
            result = document.meta[self.coarse_type]
        return result

    def generate_positive_samples(self):
        """

        :return:
        """
        events = self.events
        if self.text_encoder:
            for i, event in enumerate(events):
                mentions = event.mentions
                if len(mentions) < 2:
                    continue
                mention_keys = [self.generate_mention_key(e) for e in mentions]
                mention_embeddings = [np.expand_dims(self.mention_and_embedding[e], axis=0) for e in mention_keys]
                mention_embeddings = np.concatenate(mention_embeddings, axis=0)
                similarities = cosine_similarity(mention_embeddings)
                for j in range(len(mentions)):
                    for k in range(j + 1, len(mentions)):
                        self.positive_similarities.append(similarities[j][k])
                existing_samples = set()
                for j, mention_j in enumerate(mentions):
                    similarities_j = similarities[j]
                    least_similar_index = None
                    least_similarity = sys.maxsize
                    for k, mention_k in enumerate(mentions):
                        if k == j:
                            continue
                        if similarities_j[k] < least_similarity:
                            least_similarity = similarities_j[k]
                            least_similar_index = k
                    if (j, least_similar_index) in existing_samples:
                        continue
                    else:
                        existing_samples.add((j, least_similar_index))
                        existing_samples.add((least_similar_index, j))
                    least_similar_mention = mentions[least_similar_index]
                    self.positive_samples.append([mention_j, least_similar_mention])
        else:
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
        # result = f'{mention.doc_id}--{mention.mention_id}'
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

    def filter_negative_samples(self):
        """

        :return:
        """
        if self.text_encoder is None:
            return
        for anchor, candidates in self.negative_samples.items():
            anchor_embedding = np.expand_dims(self.mention_and_embedding[anchor], axis=0)
            candidate_embeddings = [np.expand_dims(self.mention_and_embedding[self.generate_mention_key(e)], axis=0) for e in candidates]
            candidate_embeddings = np.concatenate(candidate_embeddings, axis=0)
            similarities = cosine_similarity(anchor_embedding, candidate_embeddings).tolist()[0]
            harder_candidates = []
            for i, similarity in enumerate(similarities):
                if similarity > self.positive_similarity_median:
                    harder_candidates.append(candidates[i])
            logger.info(f'filter_negative_samples: {len(candidates)} -> {len(harder_candidates)}')
            self.negative_samples[anchor] = harder_candidates

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

            for sample in self.samples:
                first_mention, second_mention, label = sample
                first_mention_key = self.generate_mention_key(first_mention)
                first_mention_text = self.mention_and_text[first_mention_key]
                second_mention_key = self.generate_mention_key(second_mention)
                second_mention_text = self.mention_and_text[second_mention_key]
                self.samples_clean.append([first_mention_text, second_mention_text, label])
        else:
            events = self.events
            for i, event in enumerate(events):
                mentions = event.mentions
                for j, mention_j in enumerate(mentions):
                    self.samples.append([mention_j])

                    coarse_cluster_name = self.get_coarse_cluster_name(mention_j)
                    self.sample_coarse_cluster_label.append(coarse_cluster_name)

            for sample in self.samples:
                first_mention = sample[0]
                first_mention_text = self.generate_mention_text(first_mention)
                self.samples_clean.append([first_mention_text])

    def __len__(self):
        result = len(self.samples)
        return result

    def find_left_right_index(self, masks: List[float]):
        """

        :param masks:
        :return:
        """
        if 1.0 not in masks:
            print()
        left = masks.index(1.0)
        right = left
        for i in range(left + 1, len(masks)):
            if masks[i] == 1.0:
                right = i
            else:
                break
        return left, right

    def tokenize(self, texts: List[str]):
        """

        :param texts:
        :return:
        """
        left_tokens = self.tokenizer(texts[0])
        tokenized_left = [
            left_tokens['input_ids'][: -1],
            left_tokens['attention_mask'][:-1],
            # tokens['token_type_ids'],
        ]
        tokens = self.tokenizer(texts[1])
        tokenized_text = [
            tokens['input_ids'][1: -1],
            tokens['attention_mask'][1:-1],
            # tokens['token_type_ids'],
        ]
        right_tokens = self.tokenizer(texts[2])
        tokenized_right = [
            right_tokens['input_ids'][1:],
            right_tokens['attention_mask'][1:],
            # tokens['token_type_ids'],
        ]
        result = [
            tokenized_left[0] + tokenized_text[0] + tokenized_right[0],
            tokenized_left[1] + tokenized_text[1] + tokenized_right[1],
            [0.0] * len(tokenized_left[1]) + [1.0] * len(tokenized_text[1]) + [0.0] * len(tokenized_right[1]),
        ]
        result[0] = result[0][-512:]
        result[1] = result[1][-512:]
        result[2] = result[2][-512:]
        left, right = self.find_left_right_index(result[2])
        result.extend([[left], [right]])
        return result

    def __getitem__(self, idx: int):
        """

        :param idx:
        :return:
        """
        sample = self.samples_clean[idx]
        result = []
        first_tokenized_result = self.tokenize(sample[0])
        result.extend(first_tokenized_result)
        if self.train:
            result.extend(self.tokenize(sample[1]))
            result.append([sample[2]])
        else:
            result.extend(first_tokenized_result)
            result.append([0])
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


class EventCoreferenceModule(pl.LightningModule):
    """
    https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
    """
    def __init__(self, conf, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(conf)
        self.config = AutoConfig.from_pretrained(conf['transformer_model'])
        self.encoder = AutoModel.from_config(self.config)
        self.margin = 1.0
        self.validation_step_outputs = []
        self.hidden_size = self.config.hidden_size
        self.mention_mlp = nn.Linear(self.hidden_size, self.hidden_size)
        self.context_mlp = nn.Linear(self.hidden_size, self.hidden_size)
        self.pdist = nn.PairwiseDistance(p=2)

    def get_mention_representions(self, mentions, attention_masks, mention_masks):
        """

        :param mentions:
        :param attention_masks:
        :param mention_masks:
        :return:
        """
        hiddens = self.encoder(
            input_ids=mentions,
            attention_mask=attention_masks,
            # token_type_ids=token_type_ids
        )
        last_hidden_states = hiddens.last_hidden_state
        context_repr = last_hidden_states[:, 0, :]
        context_repr = self.context_mlp(context_repr)

        mention_representation = last_hidden_states * torch.unsqueeze(mention_masks, dim=2)
        mention_representation = torch.sum(mention_representation, dim=1)
        mention_representation = self.mention_mlp(mention_representation)
        repr = torch.cat([context_repr, mention_representation], dim=1)
        return repr

    def forward(self, mentions1, attention_masks1,
                mention_masks1, mention_left1, mention_right1,
                # token_type_ids1,
                mentions2, attention_masks2,
                mention_masks2, mention_left2, mention_right2,
                # token_type_ids2,
                labels,
                *args, **kwargs) \
            -> Dict[str, torch.Tensor]:
        result = {}
        mention_representations1 = self.get_mention_representions(mentions1, attention_masks1, mention_masks1)
        result['mention_representations1'] = mention_representations1
        if mentions2 is not None:
            mention_representations2 = self.get_mention_representions(mentions2, attention_masks2, mention_masks2)
            distances = 1 - torch.cosine_similarity(mention_representations1,
                                                    mention_representations2,
                                                    )
            # distances = self.pdist(mention_representations1, mention_representations2)
            distances = torch.unsqueeze(distances, dim=1)
            distances_square = torch.square(distances)
            # print()
            # print(f'label: {labels.detach().cpu().numpy()}')
            # print(f'distances: {distances.detach().cpu().numpy()}')
            # print(f'distances_square: {distances_square.detach().cpu().numpy()}')
            result['labels'] = labels
            result['distances'] = distances
            result['distances_square'] = distances_square
            one_minus_labels = 1 - labels
            loss = labels * distances_square \
                   + one_minus_labels * torch.square(torch.clamp(self.margin - distances, min=0))
            # loss = labels * distances_square + one_minus_labels * (1.0 - distances_square)
            result['instance_loss'] = loss
            loss = torch.mean(loss)
            result['loss'] = loss
        return result

    def training_step(self, batch, batch_idx):
        forward_output = self.forward(*batch[0])
        self.log('train_loss', forward_output["loss"])
        return forward_output

    def validation_step(self, batch, batch_idx: int, *args, **kwargs):
        forward_output = self.forward(*batch)
        self.log('val_loss', forward_output["loss"])
        self.validation_step_outputs.append(forward_output)
        return forward_output

    def on_validation_epoch_end(self):
        """

        :return:
        """
        instance_losses = [e['instance_loss'] for e in self.validation_step_outputs]
        all_losses = torch.cat(instance_losses)
        val_loss = torch.mean(all_losses)
        self.log('val_loss', val_loss)

        labels = torch.cat([e['labels'] for e in self.validation_step_outputs])
        distances = torch.cat([e['distances'] for e in self.validation_step_outputs])
        positive_distance = torch.sum((labels * distances)) / torch.sum(labels)
        negative_distance = torch.sum(((1 - labels) * distances)) / torch.sum((1 - labels))
        self.log('positive_distance', positive_distance)
        self.log('negative_distance', negative_distance)
        print()
        print(f'positive_distance: {positive_distance}')
        print(f'negative_distance: {negative_distance}')

        distances_square = torch.cat([e['distances_square'] for e in self.validation_step_outputs])
        positive_distance_square = torch.sum((labels * distances_square)) / torch.sum(labels)
        negative_distance_square = torch.sum(((1 - labels) * distances_square)) / torch.sum((1 - labels))
        self.log('positive_distance_square', positive_distance_square)
        self.log('negative_distance_square', negative_distance_square)
        print()
        print(f'positive_distance_square: {positive_distance_square}')
        print(f'negative_distance_square: {negative_distance_square}')

        self.validation_step_outputs.clear()

    def get_optimizer_and_scheduler(self):

        no_decay = self.hparams.no_decay_params

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        if self.hparams.optimizer == "adam":
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters, self.hparams.learning_rate)
        else:
            raise NotImplementedError

        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.hparams.num_warmup_steps,
            num_training_steps=self.hparams.num_training_steps,
        )

        return optimizer, lr_scheduler

    def configure_optimizers(self):
        optimizer, lr_scheduler = self.get_optimizer_and_scheduler()
        if lr_scheduler is None:
            return optimizer
        return [optimizer], [{"interval": "step", "scheduler": lr_scheduler}]

    def predict_step(self, batch, batch_idx: int):
        """

        :param batch:
        :param batch_idx:
        :return:
        """
        result = self(*batch)
        return result


class ContrastiveRepresentationLearningEcrModel(PlEcrModel):
    """
    Contrastive Representation Learning for Cross-Document Coreference Resolution of Events and Entities
    """

    def __init__(self, conf: DictConfig):
        super().__init__(conf)

    def build_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.conf['ecr_model']['module']['transformer_model'])

    def instanciate_module(self):
        module = EventCoreferenceModule(self.conf['ecr_model']['module'])
        return module

    def load_module(self, filepath: str):
        """

        :param filepath:
        :return:
        """
        result = EventCoreferenceModule.load_from_checkpoint(filepath)
        return result

    def prepare_data(self, data: EcrData, mode: str):
        """

        Args:
            data:
            mode:

        Returns:

        """
        if data is None:
            return None, None
        coarse_type = self.conf['topic'][f'{mode}_topic']
        dataset = EventCoreferenceDataset(data, self.tokenizer, coarse_type, train=(mode == 'train'))

        batch_size = self.conf['ecr_model']['dataloader'][f'{mode}_batch_size']
        num_workers = self.conf['ecr_model']['dataloader']['num_workers']

        dataloader = DataLoader(dataset, batch_size=batch_size,
                                collate_fn=EventCoreferenceDataset.collate_fn,
                                num_workers=num_workers,
                                shuffle=(mode == 'train')
                                )
        dataloaders = [dataloader]
        return dataset, dataloaders

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
        """

        Args:
            data:
            output_tag:

        Returns:

        """
        trainer = pl.Trainer(
            accelerator="gpu",
            # devices=args.devices
        )
        test_dataset, test_dataloaders = self.prepare_data(data, mode='predict')
        group_representations = self.inner_pred(trainer, self.module, test_dataloaders, test_dataset)
        for coarse_type, repres in group_representations.items():
            for item in repres:
                mention_repr = item[0]
                mention_id = item[1]
                data.mentions[mention_id].add_tag(output_tag, mention_repr)
        return data

    def get_predict_type(self) -> str:
        result = Mention.mention_repr_tag_name
        return result