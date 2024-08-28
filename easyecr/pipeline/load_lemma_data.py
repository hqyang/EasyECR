#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""



Date: 2023/12/16 10:40
"""
import pickle

import numpy as np

from easyecr.ecr_data.data_structure.data_structure import EcrData
from easyecr.ecr_data.data_structure.data_structure import Mention


def load_mention_map():
    now_path = "/home/nobody/project/EasyECR-lemma_ce_coref/datasets/ecb/mention_map.pkl"
    mention_map = pickle.load(open(now_path, "rb"))
    evt_mention_map = {
        m_id: m
        for m_id, m in mention_map.items()
        if m["men_type"] == "evt"
        # and m['split'] == 'test'
    }
    return evt_mention_map


def load_mention_pair_and_similarity_part(data_type: str):
    # mps, mps_trans = pickle.load(
    #     open(f"/data/dev/ecr-data/lemma_distance/ecb/lh/mp_mp_t_{data_type}.pkl", "rb")
    # )
    mps, mps_trans = pickle.load(
        open(f"/home/nobody/code/easyecr-lemma_ce_coref/datasets/gvc/lh/mp_mp_t_{data_type}.pkl", "rb")
    )

    _, _, _, fns = mps_trans
    tps, fps, tns, fns_nt = mps
    print(len(tps), len(fps), len(fns))
    # print(len(fps,))
    all_mention_pairs = tps + fps + tns + fns_nt
    similarities = np.array([1] * len(tps + fps) + [0] * len(tns + fns_nt))
    result = {}
    for i, pair in enumerate(all_mention_pairs):
        result[pair] = similarities[i]
    return result


def load_mention_pair_and_similarity():
    data_types = ["train", "dev", "test"]
    result = {}
    for data_type in data_types:
        part_result = load_mention_pair_and_similarity_part(data_type)
        result.update(part_result)
    return result


def get_mention_id(mention: Mention, data: EcrData):
    """

    Args:
        mention:
        data:

    Returns:

    """
    doc_id = mention.doc_id
    doc = data.documents[doc_id]
    doc_name = doc.meta["doc_name"]  # ecb+
    # doc_name = doc.meta["doc_title"]  # gvc
    original_mention_id = mention.get_tag("original_mention_id")  # ecb+
    # key = mention.mention_id  # gvc
    key = f"{doc_name}_{original_mention_id}"  # ecb+
    return key
