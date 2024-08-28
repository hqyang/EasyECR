#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""



Date: 2023/10/30 15:37
"""

from typing import Union
from typing import List
from typing import Dict
from typing import Optional

from sklearn.cluster import AgglomerativeClustering
import torch
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from easyecr.ecr_data.data_structure.data_structure import EcrData
from easyecr.ecr_data.data_structure.data_structure import Mention
from easyecr.ecr_model.ecr_tagger.ecr_tagger import EcrTagger
from easyecr.utils import log_utils

logger = log_utils.get_logger(__file__)


class EcrClusterModel(EcrTagger):
    def predict(
        self, data: EcrData, output_tag: str, input_tag: str, topic_tag: str = Mention.basic_topic_tag_name
    ) -> EcrData:
        """

        Args:
            data:
            output_tag:
            input_tag: repr or distance
            topic_tag:

        Returns:

        """
        raise NotImplementedError()

    def set_best_model(self, tag_with_parameters: str):
        """

        Args:
            tag_with_parameters:

        Returns:

        """
        raise NotImplementedError()


class EcrAgglomerativeClustering(EcrClusterModel):
    def __init__(
        self,
        distance_threshold: Union[float, List[float], None] = None,
        affinity: str = "cosine",
        linkage: str = "average",
        n_clusters: Optional[int] = None,
        best_distance: Optional[int] = None,
    ):
        """

        :param distance_threshold:
        :param affinity:
        :param linkage:
        :param n_clusters:
        """
        self.affinity = affinity
        self.linkage = linkage
        self.models = []
        self.distance_threshold = distance_threshold
        if distance_threshold is not None:
            if isinstance(distance_threshold, float):
                distance_threshold = [distance_threshold]
            self.distance_threshold = distance_threshold
            for threshold in distance_threshold:
                model = AgglomerativeClustering(
                    distance_threshold=threshold,
                    n_clusters=n_clusters,
                    affinity=affinity,
                    linkage=linkage,
                )
                self.models.append(model)

        self.best_model = None
        self.best_distance = best_distance
        if best_distance is not None:
            self.best_model = AgglomerativeClustering(
                distance_threshold=best_distance,
                n_clusters=n_clusters,
                affinity=affinity,
                linkage=linkage,
            )

    def predict(
        self, data: EcrData, output_tag: str, input_tag: str, topic_tag: str = Mention.basic_topic_tag_name
    ) -> EcrData:
        """

        Args:
            data:
            output_tag:
            input_tag:
            topic_tag:

        Returns:

        """
        tag_and_mentions = data.group_mention_by_tag(topic_tag)
        if self.best_model is not None:
            logger.info(f"use best model: {self.best_distance}")
            models = [self.best_model]
        else:
            models = self.models
        for i, model in enumerate(models):
            for tag, mentions in tag_and_mentions.items():
                if input_tag.startswith(Mention.mention_repr_tag_name):
                    representations = [np.expand_dims(m.get_tag(input_tag), axis=0) for m in mentions]
                    input = np.concatenate(representations, axis=0)
                elif input_tag.startswith(Mention.mention_distance_tag_name):
                    mention_ids = [e.mention_id for e in mentions]
                    input = []
                    if len(mentions) > 1:
                        for mention in mentions:
                            try:
                                mention_distances = mention.get_tag(input_tag)
                            except KeyError:
                                mention_distances = {}
                            target_mention_distances = [
                                1.0 if e == mention.mention_id or e not in mention_distances else mention_distances[e]
                                for e in mention_ids
                            ]
                            input.append(target_mention_distances)
                    else:
                        input.append([1.0])
                else:
                    raise NotImplementedError(input_tag)

                if len(input) == 1 and len(input[0]) == 1:
                    labels = [0]
                else:
                    clustering = model.fit(input)
                    labels = clustering.labels_
                for j, label in enumerate(labels):
                    if self.best_model is not None:
                        mentions[j].add_tag(output_tag, f"{tag}-{label}")
                    else:
                        # 实际的output_tag，包含额外的超惨
                        mentions[j].add_tag(f"{output_tag}|{self.distance_threshold[i]}", f"{tag}-{label}")
        return data

    def set_best_model(self, tag_with_parameters: str):
        """

        Args:
            parameters:

        Returns:

        """
        self.best_distance = float(tag_with_parameters.split("|")[1])
        self.best_model = AgglomerativeClustering(
            distance_threshold=self.best_distance,
            n_clusters=None,
            affinity=self.affinity,
            linkage=self.linkage,
        )


class EcrConnectedComponent(EcrClusterModel):
    def __init__(self, distance_threshold: Union[float, List[float], None] = None, best_distance: Optional[int] = None):
        """

        :param distance_threshold:
        """
        if isinstance(distance_threshold, float):
            distance_threshold = [distance_threshold]
        self.distance_threshold = distance_threshold
        self.best_distance = best_distance

    def cluster_cc(self, affinity_matrix, threshold=0.8):
        """
        Find connected components using the affinity matrix and threshold -> adjacency matrix
        Parameters
        ----------
        affinity_matrix: np.array
        threshold: float

        Returns
        -------
        list, np.array
        """
        adjacency_matrix = csr_matrix(affinity_matrix > threshold)
        clusters, labels = connected_components(adjacency_matrix, return_labels=True, directed=False)
        return clusters, labels

    def predict(
        self, data: EcrData, output_tag: str, input_tag: str, topic_tag: str = Mention.basic_topic_tag_name
    ) -> EcrData:
        """

        Args:
            data:
            output_tag:
            input_tag:
            topic_tag:

        Returns:

        """
        tag_and_mentions = data.group_mention_by_tag(topic_tag)
        if self.best_distance:
            logger.info(f"use best model: {self.best_distance}")
            distances = [self.best_distance]
        else:
            distances = self.distance_threshold
        for i, threshold in enumerate(distances):
            for tag, mentions in tag_and_mentions.items():
                if input_tag.startswith(Mention.mention_repr_tag_name):
                    raise NotImplementedError()
                elif input_tag.startswith(Mention.mention_distance_tag_name):
                    mention_ids = [e.mention_id for e in mentions]
                    input = []
                    if len(mentions) > 1:
                        for mention in mentions:
                            try:
                                mention_distances = mention.get_tag(input_tag)
                            except KeyError:
                                mention_distances = {}
                            target_mention_distances = [
                                1.0 if e == mention.mention_id or e not in mention_distances else mention_distances[e]
                                for e in mention_ids
                            ]
                            input.append(target_mention_distances)
                    else:
                        input.append([1.0])
                else:
                    raise NotImplementedError(input_tag)

                if len(input) == 1 and len(input[0]) == 1:
                    labels = [0]
                else:
                    input = np.array(input)
                    input = 1 - input
                    clusters, labels = self.cluster_cc(input, threshold=1 - threshold)
                for j, label in enumerate(labels):
                    if self.best_distance is not None:
                        mentions[j].add_tag(output_tag, f"{tag}-{label}")
                    else:
                        # 实际的output_tag，包含额外的超参
                        mentions[j].add_tag(f"{output_tag}|{self.distance_threshold[i]}", f"{tag}-{label}")
        return data

    def set_best_model(self, tag_with_parameters: str):
        """

        Args:
            parameters:

        Returns:

        """
        self.best_distance = float(tag_with_parameters.split("|")[1])


def cluster(representations, distance_threshold):
    """

    :param representations:
    :param distance_threshold:
    :return:
    """
    clustering = AgglomerativeClustering(
        distance_threshold=distance_threshold,
        n_clusters=None,
        affinity="cosine",
        linkage="average",
    ).fit(representations)
    result = clustering.labels_
    return result
