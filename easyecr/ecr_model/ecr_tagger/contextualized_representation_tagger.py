#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""



Date: 2023/12/6 12:48
"""
from typing import List

from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import AutoModel
import torch
from transformers import RobertaTokenizer
from transformers import RobertaModel

from easyecr.ecr_data.data_structure.data_structure import EcrData
from easyecr.ecr_data.data_structure.data_structure import Document
from easyecr.ecr_data.data_structure.data_structure import Mention
from easyecr.ecr_model.ecr_tagger.ecr_tagger import EcrTagger
from easyecr.ecr_data.data_converter.data_converter import DataConverter
from easyecr.ecr_data.datasets.wec.wec_eng import WECEngDataset


class ContextualizedRepresentationTagger(EcrTagger):
    def __init__(self, max_surrounding_contx, transformer_model, finetune=False, use_cuda=True):
        self.model = RobertaModel.from_pretrained(transformer_model)
        # self.model = BertModel.from_pretrained("bert-large-cased")
        self.tokenizer = RobertaTokenizer.from_pretrained(transformer_model)
        # self.tokenizer = BertTokenizer.from_pretrained("bert-large-cased")
        self.max_surrounding_contx = max_surrounding_contx
        self.use_cuda = use_cuda
        self.finetune = finetune
        self.embed_size = 1024
        self.cache = {}

        if self.use_cuda:
            self.model.cuda()

    def get_mention_full_rep(self, data: EcrData, mention: Mention):
        if mention.mention_id in self.cache:
            return self.cache[mention.mention_id]
        sent_ids, ment1_inx_start, ment1_inx_end, context = self.mention_feat_to_vec(data, mention)

        if self.use_cuda:
            sent_ids = sent_ids.cuda()

        if not self.finetune:
            with torch.no_grad():
                last_hidden_span = self.model(sent_ids).last_hidden_state
        else:
            last_hidden_span = self.model(sent_ids).last_hidden_state

        mention_hidden_span = last_hidden_span.view(last_hidden_span.shape[1], -1)[ment1_inx_start:ment1_inx_end]
        result = (
            mention_hidden_span,
            mention_hidden_span[0],
            mention_hidden_span[-1],
            mention_hidden_span.shape[0],
            context,
        )
        self.cache[mention.mention_id] = result
        return result

    @staticmethod
    def extract_mention_surrounding_context(data: EcrData, mention: Mention):
        context = data.get_mention_context(mention.mention_id, local_context_type="doc")[1:]
        ret_context_before, ret_mention, ret_context_after = context
        ret_context_before = ret_context_before.split(" ")
        ret_mention = ret_mention.split(" ")
        ret_context_after = ret_context_after.split(" ")

        return ret_context_before, ret_mention, ret_context_after, context

    def mention_feat_to_vec(self, data: EcrData, mention: Mention):
        (
            cntx_before_str,
            ment_span_str,
            cntx_after_str,
            context,
        ) = ContextualizedRepresentationTagger.extract_mention_surrounding_context(data, mention)

        cntx_before, cntx_after = cntx_before_str, cntx_after_str
        if len(cntx_before_str) != 0:
            cntx_before = self.tokenizer.encode(" ".join(cntx_before_str), add_special_tokens=False)
        if len(cntx_after_str) != 0:
            cntx_after = self.tokenizer.encode(" ".join(cntx_after_str), add_special_tokens=False)

        if self.max_surrounding_contx != -1:
            if len(cntx_before) > self.max_surrounding_contx:
                cntx_before = cntx_before[-self.max_surrounding_contx + 1 :]
            if len(cntx_after) > self.max_surrounding_contx:
                cntx_after = cntx_after[: self.max_surrounding_contx - 1]

        ment_span = self.tokenizer.encode(" ".join(ment_span_str), add_special_tokens=False)

        if isinstance(ment_span, torch.Tensor):
            ment_span = ment_span.tolist()
        if isinstance(cntx_before, torch.Tensor):
            cntx_before = cntx_before.tolist()
        if isinstance(cntx_after, torch.Tensor):
            cntx_after = cntx_after.tolist()

        all_sent_toks = [[0] + cntx_before + ment_span + cntx_after + [2]]
        sent_tokens = torch.tensor(all_sent_toks)
        mention_start_idx = len(cntx_before) + 1
        mention_end_idx = len(cntx_before) + len(ment_span) + 1
        assert all_sent_toks[0][mention_start_idx:mention_end_idx] == ment_span
        return sent_tokens, mention_start_idx, mention_end_idx, context

    def get_embed_size(self):
        return self.embed_size

    def get_mentions_rep(self, mentions_list):
        embed_list = [self.get_mention_full_rep(mention) for mention in mentions_list]
        return embed_list

    def predict(self, data: EcrData, output_tag: str) -> EcrData:
        """

        Args:
            data:
            output_tag:

        Returns:

        """
        for mention_id, mention in data.mentions.items():
            (
                mention_hiddens,
                mention_hiddens_first,
                mention_hiddens_last,
                mention_length,
                context,
            ) = self.get_mention_full_rep(data, mention)

            mention.meta[f"{output_tag}_hiddens"] = mention_hiddens.cpu().numpy()
            mention.meta[f"{output_tag}_length"] = mention_length
            mention.meta[f"{output_tag}_hiddens_first"] = mention_hiddens_first.cpu().numpy()
            mention.meta[f"{output_tag}_hiddens_last"] = mention_hiddens_last.cpu().numpy()
            mention.meta[f"{output_tag}_context"] = context
        return data


if __name__ == "__main__":
    dataset_name = "ecbplus"
    filepath = "/home/nobody/project/ecr-data/data/ECB+_LREC2014_test/raw_data"
    data = DataConverter.from_directory(dataset_name, filepath)
    tagger = ContextualizedRepresentationTagger(transformer_model="roberta-base")
    tagged_data = tagger.predict(data, "event_id_pred")
    print("end")

    # data = WECEngDataset(directory='/data/dev/ecr-data/WEC-Eng', selection='Dev').to_ecr_data()
    # tagger = ContextualizedRepresentationTagger(transformer_model='roberta-base')
    # tagged_data = tagger.predict(data, output_tag='mention')
    # print('end')
