#!/usr/bin/env python
# -*- coding:utf-8 -*-　
"""



Date: 2023/10/30 17:52
"""
from collections import defaultdict


def group_elements(elements: list, cluster_labels):
    """

    :param elements:
    :param cluster_labels:
    :return:
    """
    group_dict = defaultdict(list)
    for i, label in enumerate(cluster_labels):
        group_dict[label].append(elements[i])
    result = [e for e in group_dict.values()]
    return result
