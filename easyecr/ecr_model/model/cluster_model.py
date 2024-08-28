#!/usr/bin/env python
# -*- coding:utf-8 -*-　
"""



Date: 2023/10/30 15:37
"""

from sklearn.cluster import AgglomerativeClustering


class ClusterModel:
    pass


def cluster(representations, distance_threshold):
    """

    :param representations:
    :param distance_threshold:
    :return:
    """
    clustering = AgglomerativeClustering(distance_threshold=distance_threshold, n_clusters=None,
                                         affinity='cosine',
                                         linkage="average",
                                         ).fit(representations)
    result = clustering.labels_
    return result
