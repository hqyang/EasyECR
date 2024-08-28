#!/usr/bin/env python
# -*- coding:utf-8 -*-　
"""



Date: 2023/12/14 17:41
"""
from typing import List
from typing import Tuple
from typing import Set

from omegaconf import DictConfig
import spacy
from spacy.tokens import Doc
from openai import OpenAI

from easyecr.ecr_model.model.simple_ecr_models.simple_ecr_models import SimpleDistanceEcrModel
from easyecr.ecr_data.data_structure.data_structure import EcrData
from easyecr.ecr_data.data_structure.data_structure import Mention
from easyecr.utils import chatgpt_client


class ChatGPTDistance(SimpleDistanceEcrModel):
    def __init__(self, train_topic: str, predict_topic: str, model: str, api_key: str):
        """

        Args:
            conf:
        """
        super().__init__(train_topic, predict_topic)
        # defaults to os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def train(self, train_data: EcrData, dev_data: EcrData):
        """

        Args:
            train_data:
            dev_data:

        Returns:

        """
        pass

    def mask_context(self, context: Tuple[str, str, str]) -> str:
        """

        Args:
            context:

        Returns:

        """
        parts = [context[0], '<m>', context[1], '</m>', context[2]]
        result = ' '.join(parts)
        return result

    def predict_mention_distance(self, mention1: Mention, mention2: Mention, data: EcrData) -> float:
        """

        Args:
            mention1:
            mention2:
            data:

        Returns:

        """
        context1 = data.get_mention_context(mention1.mention_id)
        context2 = data.get_mention_context(mention2.mention_id)

        context1 = self.mask_context(context1)
        context2 = self.mask_context(context2)
        prompt = f"""
        下面文本中通过<m>和</m>标记了两个事件，请判断这两个事件是否指的是现实世界中的同一个事件。是现实世界中的同一个事件返回，是，否则返回，否。
        文本: {context1} {context2}
        是否共指：
        """
        messages = [
            {
                "role": "system",
                "content": "You are an ."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        result = chatgpt_client.simple_chat(messages, self.client, model=self.model)
        if result == '是':
            distance = 0
        else:
            distance = 1
        return distance


