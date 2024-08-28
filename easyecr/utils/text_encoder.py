#!/usr/bin/env python
# -*- coding:utf-8 -*-　
"""



"""
from typing import List
import os

import torch
from transformers import AutoModel, AutoTokenizer
from scipy.spatial.distance import cosine


class TextEncoder:
    def __init__(self, model_name: str = 'sgpt', version='1.3b', device=0):
        """

        :param model_name:
        """
        if model_name == 'sgpt':
            self.model = self.init_sgpt(device=device, version=version)
        else:
            raise NotImplementedError
        self.cache = {}

    def init_sgpt(self, device=0, version='1.3b'):
        assert version in {'5b', '1.3b', '125m'}, 'invalid version'
        tokenizer = AutoTokenizer.from_pretrained(
            f'/cto_studio/LINSIDA_left_behind/linsida/.cache/huggingface/hub/sgpt_{version}')
        model = AutoModel.from_pretrained(
            f'/cto_studio/LINSIDA_left_behind/linsida/.cache/huggingface/hub/sgpt_{version}')
        model.eval()
        model.to(device)
        model.tokenizer = tokenizer
        return model

    def encode(self, texts: List[str], max_text_len: int = 48):
        """

        :param text:
        :param max_text_len: 超过长度的文本会被截断
        :return:
        """
        texts = [text if len(text) < max_text_len else text[:max_text_len] for text in texts]
        device = next(self.model.children()).weight.device
        batch_tokens = self.model.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        batch_tokens = batch_tokens.to(device)

        # Get the embeddings
        with torch.no_grad():
            # Get hidden state of shape [bs, seq_len, hid_dim]
            last_hidden_state = self.model(**batch_tokens, output_hidden_states=True, return_dict=True).last_hidden_state

        # Get weights of shape [bs, seq_len, hid_dim]
        weights = (
            torch.arange(start=1, end=last_hidden_state.shape[1] + 1)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(last_hidden_state.size())
            .float().to(last_hidden_state.device)
        )

        # Get attn mask of shape [bs, seq_len, hid_dim]
        input_mask_expanded = (
            batch_tokens["attention_mask"]
            .unsqueeze(-1)
            .expand(last_hidden_state.size())
            .float()
        )
        # Perform weighted mean pooling across seq_len: bs, seq_len, hidden_dim -> bs, hidden_dim
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded * weights, dim=1)
        sum_mask = torch.sum(input_mask_expanded * weights, dim=1)
        embeddings = sum_embeddings / sum_mask
        result = embeddings.cpu()

        del batch_tokens
        torch.cuda.empty_cache()

        return result

    def compute_semantic_similarity(self, texts: List[str]):
        """
        :param texts:
        :return:
        """
        embeddings = self.encode(texts)
        similarity = 1 - cosine(embeddings[0], embeddings[1])
        return similarity

    def compute_cosine_similarity(self, embeddings: List[str]):
        """
        :param texts:
        :return:
        """
        similarity = 1 - cosine(embeddings[0], embeddings[1])
        return similarity


if __name__ == '__main__':
    text_encoder = TextEncoder(device=6, version='1.3b')
    texts = [
        'http : / / www . espn . co . uk / boxing / sport / story / 159389 . html Klitschko stops Thompson in regulation win Wladimir Klitschko retained his IBF , WBO , WBA & IBO heavyweight world titles with a second professional <trigger>victory</trigger> over Tony Thompson on Saturday .',

        'http : / / www . espn . co . uk / boxing / sport / story / 159389 . html Klitschko stops Thompson in regulation win Klitschko stops Thompson in regulation <trigger>win</trigger>',

        'http : / / www . bbc . co . uk / sport / 0 / boxing / 18747023 Wladimir Klitschko defeats Tony Thompson in six rounds Wladimir Klitschko <trigger>defeats</trigger> Tony Thompson in six rounds'
    ]
    embeddings = text_encoder.encode(texts)
    print()
