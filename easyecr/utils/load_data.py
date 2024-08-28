#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""



Date: 2023/12/25 10:28
"""
import os

from easyecr.ecr_data.data_converter.data_converter import SplitDataConverter
from easyecr.utils import object_utils


def get_dataset(
    dataset_name: str,
    cache_dir: str = "/home/nobody/project/EasyECR/cache",
    train_path: str = "",
    dev_path: str = "",
    test_path: str = "",
    total_path: str = "",
):
    """

    Args:
        config_name:
        use_cache:
        cache_dir:
        total_path:

    Returns:

    """
    os.makedirs(cache_dir, exist_ok=True)
    train_data_cache_path = os.path.join(cache_dir, f"{dataset_name}.train.pkl")
    dev_data_cache_path = os.path.join(cache_dir, f"{dataset_name}.dev.pkl")
    test_data_cache_path = os.path.join(cache_dir, f"{dataset_name}.test.pkl")
    if (
        os.path.exists(train_data_cache_path)
        and os.path.exists(dev_data_cache_path)
        and os.path.exists(test_data_cache_path)
    ):
        train_data = object_utils.load(train_data_cache_path)
        dev_data = object_utils.load(dev_data_cache_path)
        test_data = object_utils.load(test_data_cache_path)
    else:
        if total_path:
            train_data, dev_data, test_data = SplitDataConverter.split(dataset_name, total_path=total_path)
        else:
            train_data, dev_data, test_data = SplitDataConverter.split(
                dataset_name, train_path=train_path, dev_path=dev_path, test_path=test_path
            )

        object_utils.save(train_data, train_data_cache_path)
        object_utils.save(dev_data, dev_data_cache_path)
        object_utils.save(test_data, test_data_cache_path)
    return train_data, dev_data, test_data
