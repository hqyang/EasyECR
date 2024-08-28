#!/usr/bin/env python
# -*- coding:utf-8 -*-　
"""



Date: 2023/12/18 12:56
"""
from typing import List

from scipy.spatial.distance import cosine
from transformers import AutoModel
from transformers import AutoTokenizer
import torch

from easyecr.ecr_model.model.simple_ecr_models.simple_ecr_models import SimpleReprEcrModel
from easyecr.ecr_data.data_structure.data_structure import EcrData
from easyecr.ecr_data.data_structure.data_structure import Mention


class SgptTextEncoderRepr(SimpleReprEcrModel):
    """

    """
    def __init__(self, version='125m', device=0):
        """

        Args:
            version:
            device:
        """
        self.version = version
        self.device = device
        self.model = self.init_sgpt(device=device, version=version)
        self.cache = {}

    def init_sgpt(self, device, version):
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
        result = embeddings.cpu().numpy()

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

    def predict_mention_repr(self, mention: Mention, data: EcrData) -> List[float]:
        """

        Args:
            mention:
            data:

        Returns:

        """
        context1 = mention.extent.text
        if context1 in self.cache:
            embedding = self.cache[context1]
        else:
            embedding = self.encode([context1])[0]
            self.cache[context1] = embedding
        return embedding
