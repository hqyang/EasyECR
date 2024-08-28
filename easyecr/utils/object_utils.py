#!/usr/bin/env python
# -*- coding:utf-8 -*-　
"""



Date: 2023/10/26 17:10
"""
import pickle


def save(obj, filepath: str):
    """

    :param obj:
    :param filepath:
    :return:
    """
    with open(filepath, mode='wb') as output_file:
        pickle.dump(obj, output_file)


def load(filepath: str):
    """

    :param filepath:
    :return:
    """
    with open(filepath, mode='rb') as input_file:
        data = pickle.load(input_file)
    return data


def convert_to_dict(obj):
    """
    Recursively converts a Python object into a dictionary.
    """
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        # 基本数据类型和None直接返回
        return obj
    elif isinstance(obj, dict):
        # 对字典的每个键值对进行递归处理
        return {k: convert_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        # 对可迭代的对象进行递归处理
        return [convert_to_dict(item) for item in obj]
    elif hasattr(obj, "__dict__"):
        # 如果对象有__dict__属性，说明它可能是一个自定义的类实例，尝试递归转换它的属性
        return {k: convert_to_dict(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
    else:
        # 如果无法识别的类型，尝试直接转换为字符串
        return str(obj)