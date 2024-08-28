#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""



Date: 2023/11/23 10:15
"""
from easyecr.ecr_data.data_structure.data_structure import EcrData


class BaseEcrDataset:
    """ """

    def __init__(self, dataset_name: str, directory: str):
        """

        :param dataset_name:
        :param directory:
        """
        self.dataset_name = dataset_name
        self.directory = directory

    @staticmethod
    def read_all_content(filepath, encoding="utf-8", keep_line_separator=False):
        """

        :param filepath:
        :param encoding:
        :return:
        """
        new_line = None
        if keep_line_separator:
            new_line = ""
        with open(filepath, encoding=encoding, newline=new_line) as in_file:
            return in_file.read()

    def to_ecr_data(self) -> "EcrData":
        """

        :return:
        """
        raise NotImplementedError()
