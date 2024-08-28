#!/usr/bin/env python
# -*- coding:utf-8 -*-　
"""



Date: 2023/12/6 12:58
"""

from easyecr.ecr_data.data_structure.data_structure import EcrData


class EcrTagger:
    def predict(self, data: EcrData, output_tag: str, **kwargs) -> EcrData:
        """

        Args:
            data:
            output_tag:
            kwargs:
                input_tag 有的tagger作用在整个EcrData数据上，不需要input_tag，但是有的tagger作用在
                mention的某个tag对应的数据上，需要input_tag

        Returns:

        """
        raise NotImplementedError()
