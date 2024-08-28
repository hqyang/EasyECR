import json
from typing import Any
from typing import Dict
from typing import Optional
import random
from collections import defaultdict

import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AdamW
from omegaconf import DictConfig
from transformers import AutoConfig
from torch.utils.data import Dataset
from transformers import RobertaModel
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer
from transformers import get_scheduler
from torch.utils.data import DataLoader
from transformers.activations import gelu
from transformers import RobertaTokenizer
from transformers import RobertaPreTrainedModel
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor
from pytorch_lightning import seed_everything

from easyecr.ecr_data.data_structure.data_structure import EcrData
from easyecr.ecr_data.data_structure.data_structure import Mention
from easyecr.ecr_data.data_structure.data_structure import Event
from easyecr.ecr_model.model.pl_ecr_models.pl_ecr_model import PlEcrModel

PROMPT_TYPE = [
    "hn",
    "hc",
    "hq",  # base prompts
    "sn",
    "sc",
    "sq",
    "m_ht_hn",
    "m_ht_hc",
    "m_ht_hq",
    "m_hta_hn",
    "m_hta_hc",
    "m_hta_hq",
    "m_htao_hn",
    "m_htao_hc",
    "m_htao_hq",  # mix prompts
    "m_st_hn",
    "m_st_hc",
    "m_st_hq",
    "m_sta_hn",
    "m_sta_hc",
    "m_sta_hq",
    "m_stao_hn",
    "m_stao_hc",
    "m_stao_hq",
    "ma_remove-prefix",
    "ma_remove-anchor",
    "ma_remove-match",
    "ma_remove-subtype-match",
    "ma_remove-arg-match",  # mix prompt m_hta_hn ablations
]
WORD_FILTER = set(
    [
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "i",
        "me",
        "my",
        "mine",
        "myself",
        "we",
        "us",
        "our",
        "ours",
        "ourselves",
        "he",
        "his",
        "him",
        "himself",
        "she",
        "her",
        "herself",
        "hers",
        "it",
        "its",
        "itself",
        "they",
        "their",
        "theirs",
        "them",
        "themselves",
        "other",
        "others",
        "this",
        "that",
        "these",
        "those",
        "who",
        "whom",
        "what",
        "whose",
        "which",
        "where",
        "why",
        "that",
        "all",
        "each",
        "either",
        "neither",
        "one",
        "any",
        "oneself",
        "such",
        "same",
        "everyone",
        "anyone",
        "there",
    ]
)
SELECT_ARG_STRATEGY = ["no_filter", "filter_related_args", "filter_all"]
EVENT_SUBTYPES = [  # 18 subtypes
    "artifact",
    "transferownership",
    "transaction",
    "broadcast",
    "contact",
    "demonstrate",
    "injure",
    "transfermoney",
    "transportartifact",
    "attack",
    "meet",
    "elect",
    "endposition",
    "correspondence",
    "arrestjail",
    "startposition",
    "transportperson",
    "die",
]
id2subtype = {idx: c for idx, c in enumerate(EVENT_SUBTYPES, start=1)}
id2subtype[0] = "other"
subtype2id = {v: k for k, v in id2subtype.items()}


def get_pred_related_info(simi_file: str):
    related_info_dict = {}
    with open(simi_file, "rt", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line.strip())
            related_info_dict[sample["doc_id"]] = {
                int(offset): {
                    "arguments": related_info["arguments"],
                    "related_triggers": related_info["related_triggers"],
                    "related_arguments": related_info["related_arguments"],
                }
                for offset, related_info in sample["relate_info"].items()
            }
    return related_info_dict


def create_base_template(e1_trigger: str, e2_trigger: str, prompt_type: str, s_tokens: dict) -> dict:
    trigger_offsets = []
    if prompt_type.startswith("h"):  # hard template
        if prompt_type == "hn":
            template = f"In the following text, events expressed by {s_tokens['e1s']} "
            trigger_offsets.append([len(template), len(template) + len(e1_trigger) - 1])
            template += f"{e1_trigger} {s_tokens['e1e']} and {s_tokens['e2s']} "
            trigger_offsets.append([len(template), len(template) + len(e2_trigger) - 1])
            template += f"{e2_trigger} {s_tokens['e2e']} refer to {s_tokens['mask']} event: "
        elif prompt_type == "hc":
            template = f"In the following text, the event expressed by {s_tokens['e1s']} "
            trigger_offsets.append([len(template), len(template) + len(e1_trigger) - 1])
            template += f"{e1_trigger} {s_tokens['e1e']} {s_tokens['mask']} the event expressed by {s_tokens['e2s']} "
            trigger_offsets.append([len(template), len(template) + len(e2_trigger) - 1])
            template += f"{e2_trigger} {s_tokens['e2e']}: "
        elif prompt_type == "hq":
            template = f"In the following text, do events expressed by {s_tokens['e1s']} "
            trigger_offsets.append([len(template), len(template) + len(e1_trigger) - 1])
            template += f"{e1_trigger} {s_tokens['e1e']} and {s_tokens['e2s']} "
            trigger_offsets.append([len(template), len(template) + len(e2_trigger) - 1])
            template += f"{e2_trigger} {s_tokens['e2e']} refer to the same event? {s_tokens['mask']}. "
        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
    elif prompt_type.startswith("s"):  # soft template
        if prompt_type == "sn":
            template = f"In the following text, {s_tokens['l1']} {s_tokens['e1s']} "
            trigger_offsets.append([len(template), len(template) + len(e1_trigger) - 1])
            template += f"{e1_trigger} {s_tokens['e1e']} {s_tokens['l2']} {s_tokens['l3']} {s_tokens['e2s']} "
            trigger_offsets.append([len(template), len(template) + len(e2_trigger) - 1])
            template += f"{e2_trigger} {s_tokens['e2e']} {s_tokens['l4']} {s_tokens['l5']} {s_tokens['mask']} {s_tokens['l6']}: "
        elif prompt_type == "sc":
            template = f"In the following text, {s_tokens['l1']} {s_tokens['e1s']} "
            trigger_offsets.append([len(template), len(template) + len(e1_trigger) - 1])
            template += f"{e1_trigger} {s_tokens['e1e']} {s_tokens['l2']} "
            template += f"{s_tokens['l5']} {s_tokens['mask']} {s_tokens['l6']} {s_tokens['l3']} {s_tokens['e2s']} "
            trigger_offsets.append([len(template), len(template) + len(e2_trigger) - 1])
            template += f"{e2_trigger} {s_tokens['e2e']} {s_tokens['l4']}: "
        elif prompt_type == "sq":
            template = f"In the following text, {s_tokens['l1']} {s_tokens['e1s']} "
            trigger_offsets.append([len(template), len(template) + len(e1_trigger) - 1])
            template += f"{e1_trigger} {s_tokens['e1e']} {s_tokens['l2']} {s_tokens['l3']} {s_tokens['e2s']} "
            trigger_offsets.append([len(template), len(template) + len(e2_trigger) - 1])
            template += f"{e2_trigger} {s_tokens['e2e']} {s_tokens['l4']}? {s_tokens['mask']}. "
        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    special_tokens = (
        [s_tokens["e1s"], s_tokens["e1e"], s_tokens["e2s"], s_tokens["e2e"]]
        if prompt_type.startswith("h")
        else [
            s_tokens["e1s"],
            s_tokens["e1e"],
            s_tokens["e2s"],
            s_tokens["e2e"],
            s_tokens["l1"],
            s_tokens["l2"],
            s_tokens["l3"],
            s_tokens["l4"],
            s_tokens["l5"],
            s_tokens["l6"],
        ]
    )
    if "c" in prompt_type:  # connect template
        special_tokens += [s_tokens["refer"], s_tokens["no_refer"]]
    return {"template": template, "trigger_offsets": trigger_offsets, "special_tokens": special_tokens}


def create_event_context(
    e1_sent_idx: int,
    e1_sent_start: int,
    e1_trigger: str,
    e2_sent_idx: int,
    e2_sent_start: int,
    e2_trigger: str,
    sentences: list,
    sentence_lens: list,
    s_tokens: dict,
    tokenizer,
    max_length: int,
) -> dict:
    """create segments contains events
    # Args
    [e1/e2_]e_sent_idx:
        host sentence index
    [e1/e2_]e_sent_start:
        trigger offset in the host sentence
    [e1/e2_]e_trigger:
        trigger of the event
    sentences:
        all the sentences in the document, {"start": sentence offset in the document, "text": content}
    sentence_lens:
        token numbers of all the sentences (not include [CLS], [SEP], etc.)
    s_tokens:
        special token dictionary
    tokenizer:
        tokenizer of the chosen PTM
    max_length:
        max total token numbers of segments (not include [CLS], [SEP], etc.)
    # Return
    type:
        context type, 'same_sent' or 'diff_sent', two events in the same/different sentence
    [e1/e2_]core_context:
        the host sentence contains the event
    [e1/e2_]before_context:
        context before the host sentence
    [e1/e2_]after_context:
        context after the host sentence
    e1s_core_offset, e1e_core_offset, e2s_core_offset, e2e_core_offset:
        offsets of triggers in the host sentence
    """
    if e1_sent_idx == e2_sent_idx:  # two events in the same sentence
        assert e1_sent_start < e2_sent_start
        e1_e2_sent = sentences[e1_sent_idx]["text"]
        core_context_before = f"{e1_e2_sent[:e1_sent_start]}"
        core_context_after = f"{e1_e2_sent[e2_sent_start + len(e2_trigger):]}"
        e1s_offset = 0
        core_context_middle = f"{s_tokens['e1s']} {e1_trigger} "
        e1e_offset = len(core_context_middle)
        core_context_middle += f"{s_tokens['e1e']}{e1_e2_sent[e1_sent_start + len(e1_trigger):e2_sent_start]}"
        e2s_offset = len(core_context_middle)
        core_context_middle += f"{s_tokens['e2s']} {e2_trigger} "
        e2e_offset = len(core_context_middle)
        core_context_middle += f"{s_tokens['e2e']}"
        # segment contain the two events
        core_context = core_context_before + core_context_middle + core_context_after
        total_length = len(tokenizer.tokenize(core_context))
        before_context, after_context = "", ""
        if total_length > max_length:  # cut segment
            before_after_length = (max_length - len(tokenizer.tokenize(core_context_middle))) // 2
            core_context_before = tokenizer.decode(tokenizer.encode(core_context_before)[1:-1][-before_after_length:])
            core_context_after = tokenizer.decode(tokenizer.encode(core_context_after)[1:-1][:before_after_length])
            core_context = core_context_before + core_context_middle + core_context_after
            e1s_offset, e1e_offset, e2s_offset, e2e_offset = np.asarray(
                [e1s_offset, e1e_offset, e2s_offset, e2e_offset]
            ) + np.full((4,), len(core_context_before))
        else:  # create contexts before/after the host sentence
            e1s_offset, e1e_offset, e2s_offset, e2e_offset = np.asarray(
                [e1s_offset, e1e_offset, e2s_offset, e2e_offset]
            ) + np.full((4,), len(core_context_before))
            e_before, e_after = e1_sent_idx - 1, e1_sent_idx + 1
            while True:
                if e_before >= 0:
                    if total_length + sentence_lens[e_before] <= max_length:
                        before_context = sentences[e_before]["text"] + " " + before_context
                        total_length += 1 + sentence_lens[e_before]
                        e_before -= 1
                    else:
                        e_before = -1
                if e_after < len(sentences):
                    if total_length + sentence_lens[e_after] <= max_length:
                        after_context += " " + sentences[e_after]["text"]
                        total_length += 1 + sentence_lens[e_after]
                        e_after += 1
                    else:
                        e_after = len(sentences)
                if e_before == -1 and e_after == len(sentences):
                    break
        tri1s_core_offset, tri1e_core_offset = e1s_offset + len(s_tokens["e1s"]) + 1, e1e_offset - 2
        tri2s_core_offset, tri2e_core_offset = e2s_offset + len(s_tokens["e2s"]) + 1, e2e_offset - 2
        assert core_context[e1s_offset:e1e_offset] == s_tokens["e1s"] + " " + e1_trigger + " "
        assert core_context[e1e_offset : e1e_offset + len(s_tokens["e1e"])] == s_tokens["e1e"]
        assert core_context[e2s_offset:e2e_offset] == s_tokens["e2s"] + " " + e2_trigger + " "
        assert core_context[e2e_offset : e2e_offset + len(s_tokens["e2e"])] == s_tokens["e2e"]
        assert core_context[tri1s_core_offset : tri1e_core_offset + 1] == e1_trigger
        assert core_context[tri2s_core_offset : tri2e_core_offset + 1] == e2_trigger
        return {
            "type": "same_sent",
            "core_context": core_context,
            "before_context": before_context,
            "after_context": after_context,
            "e1s_core_offset": e1s_offset,
            "e1e_core_offset": e1e_offset,
            "tri1s_core_offset": tri1s_core_offset,
            "tri1e_core_offset": tri1e_core_offset,
            "e2s_core_offset": e2s_offset,
            "e2e_core_offset": e2e_offset,
            "tri2s_core_offset": tri2s_core_offset,
            "tri2e_core_offset": tri2e_core_offset,
        }
    else:  # two events in different sentences
        e1_sent, e2_sent = sentences[e1_sent_idx]["text"], sentences[e2_sent_idx]["text"]
        # e1 source sentence
        e1_core_context_before = f"{e1_sent[:e1_sent_start]}"
        e1_core_context_after = f"{e1_sent[e1_sent_start + len(e1_trigger):]}"
        e1s_offset = 0
        e1_core_context_middle = f"{s_tokens['e1s']} {e1_trigger} "
        e1e_offset = len(e1_core_context_middle)
        e1_core_context_middle += f"{s_tokens['e1e']}"
        # e2 source sentence
        e2_core_context_before = f"{e2_sent[:e2_sent_start]}"
        e2_core_context_after = f"{e2_sent[e2_sent_start + len(e2_trigger):]}"
        e2s_offset = 0
        e2_core_context_middle = f"{s_tokens['e2s']} {e2_trigger} "
        e2e_offset = len(e2_core_context_middle)
        e2_core_context_middle += f"{s_tokens['e2e']}"
        # segment contain the two events
        e1_core_context = e1_core_context_before + e1_core_context_middle + e1_core_context_after
        e2_core_context = e2_core_context_before + e2_core_context_middle + e2_core_context_after
        total_length = len(tokenizer.tokenize(e1_core_context)) + len(tokenizer.tokenize(e2_core_context))
        e1_before_context, e1_after_context, e2_before_context, e2_after_context = "", "", "", ""
        if total_length > max_length:
            e1_e2_middle_length = len(tokenizer.tokenize(e1_core_context_middle)) + len(
                tokenizer.tokenize(e2_core_context_middle)
            )
            before_after_length = (max_length - e1_e2_middle_length) // 4
            e1_core_context_before = tokenizer.decode(
                tokenizer.encode(e1_core_context_before)[1:-1][-before_after_length:]
            )
            e1_core_context_after = tokenizer.decode(
                tokenizer.encode(e1_core_context_after)[1:-1][:before_after_length]
            )
            e1_core_context = e1_core_context_before + e1_core_context_middle + e1_core_context_after
            e1s_offset, e1e_offset = np.asarray([e1s_offset, e1e_offset]) + np.full((2,), len(e1_core_context_before))
            e2_core_context_before = tokenizer.decode(
                tokenizer.encode(e2_core_context_before)[1:-1][-before_after_length:]
            )
            e2_core_context_after = tokenizer.decode(
                tokenizer.encode(e2_core_context_after)[1:-1][:before_after_length]
            )
            e2_core_context = e2_core_context_before + e2_core_context_middle + e2_core_context_after
            e2s_offset, e2e_offset = np.asarray([e2s_offset, e2e_offset]) + np.full((2,), len(e2_core_context_before))
        else:  # add other sentences
            e1s_offset, e1e_offset = np.asarray([e1s_offset, e1e_offset]) + np.full((2,), len(e1_core_context_before))
            e2s_offset, e2e_offset = np.asarray([e2s_offset, e2e_offset]) + np.full((2,), len(e2_core_context_before))
            e1_before, e1_after, e2_before, e2_after = (
                e1_sent_idx - 1,
                e1_sent_idx + 1,
                e2_sent_idx - 1,
                e2_sent_idx + 1,
            )
            while True:
                e1_after_dead, e2_before_dead = False, False
                if e1_before >= 0:
                    if total_length + sentence_lens[e1_before] <= max_length:
                        e1_before_context = sentences[e1_before]["text"] + " " + e1_before_context
                        total_length += 1 + sentence_lens[e1_before]
                        e1_before -= 1
                    else:
                        e1_before = -1
                if e1_after <= e2_before:
                    if total_length + sentence_lens[e1_after] <= max_length:
                        e1_after_context += " " + sentences[e1_after]["text"]
                        total_length += 1 + sentence_lens[e1_after]
                        e1_after += 1
                    else:
                        e1_after_dead = True
                if e2_before >= e1_after:
                    if total_length + sentence_lens[e2_before] <= max_length:
                        e2_before_context = sentences[e2_before]["text"] + " " + e2_before_context
                        total_length += 1 + sentence_lens[e2_before]
                        e2_before -= 1
                    else:
                        e2_before_dead = True
                if e2_after < len(sentences):
                    if total_length + sentence_lens[e2_after] <= max_length:
                        e2_after_context += " " + sentences[e2_after]["text"]
                        total_length += 1 + sentence_lens[e2_after]
                        e2_after += 1
                    else:
                        e2_after = len(sentences)
                if (
                    e1_before == -1
                    and e2_after == len(sentences)
                    and ((e1_after_dead and e2_before_dead) or e1_after > e2_before)
                ):
                    break
        tri1s_core_offset, tri1e_core_offset = e1s_offset + len(s_tokens["e1s"]) + 1, e1e_offset - 2
        tri2s_core_offset, tri2e_core_offset = e2s_offset + len(s_tokens["e2s"]) + 1, e2e_offset - 2
        assert e1_core_context[e1s_offset:e1e_offset] == s_tokens["e1s"] + " " + e1_trigger + " "
        assert e1_core_context[e1e_offset : e1e_offset + len(s_tokens["e1e"])] == s_tokens["e1e"]
        assert e2_core_context[e2s_offset:e2e_offset] == s_tokens["e2s"] + " " + e2_trigger + " "
        assert e2_core_context[e2e_offset : e2e_offset + len(s_tokens["e2e"])] == s_tokens["e2e"]
        assert e1_core_context[tri1s_core_offset : tri1e_core_offset + 1] == e1_trigger
        assert e2_core_context[tri2s_core_offset : tri2e_core_offset + 1] == e2_trigger
        return {
            "type": "diff_sent",
            "e1_core_context": e1_core_context,
            "e1_before_context": e1_before_context,
            "e1_after_context": e1_after_context,
            "e1s_core_offset": e1s_offset,
            "e1e_core_offset": e1e_offset,
            "tri1s_core_offset": tri1s_core_offset,
            "tri1e_core_offset": tri1e_core_offset,
            "e2_core_context": e2_core_context,
            "e2_before_context": e2_before_context,
            "e2_after_context": e2_after_context,
            "e2s_core_offset": e2s_offset,
            "e2e_core_offset": e2e_offset,
            "tri2s_core_offset": tri2s_core_offset,
            "tri2e_core_offset": tri2e_core_offset,
        }


def create_mix_template(
    e1_trigger: str,
    e2_trigger: str,
    e1_arg_str: str,
    e2_arg_str: str,
    e1_related_str: str,
    e2_related_str: str,
    prompt_type: str,
    s_tokens: dict,
) -> dict:
    remove_prefix_temp, remove_anchor_temp = False, False
    remove_match, remove_subtype_match, remove_arg_match = False, False, False
    if prompt_type.startswith("ma"):  # m_hta_hn prompt ablation
        anchor_temp_type, inference_temp_type = "hta", "hn"
        ablation = prompt_type.split("_")[1]
        if ablation == "remove-prefix":
            remove_prefix_temp = True
        elif ablation == "remove-anchor":
            remove_anchor_temp = True
        elif ablation == "remove-match":
            remove_match = True
        elif ablation == "remove-subtype-match":
            remove_subtype_match = True
        elif ablation == "remove-arg-match":
            remove_arg_match = True
    else:
        _, anchor_temp_type, inference_temp_type = prompt_type.split("_")
    # prefix template
    prefix_trigger_offsets = []
    prefix_template = f"In the following text, the focus is on the events expressed by {s_tokens['e1s']} "
    prefix_trigger_offsets.append([len(prefix_template), len(prefix_template) + len(e1_trigger) - 1])
    prefix_template += f"{e1_trigger} {s_tokens['e1e']} and {s_tokens['e2s']} "
    prefix_trigger_offsets.append([len(prefix_template), len(prefix_template) + len(e2_trigger) - 1])
    prefix_template += (
        f"{e2_trigger} {s_tokens['e2e']}, and it needs to judge whether they refer to the same or different events: "
    )
    # anchor template
    if anchor_temp_type.startswith("h"):  # hard template
        e1_anchor_temp = "Here "
        e1s_anchor_offset = len(e1_anchor_temp)
        e1_anchor_temp += f"{s_tokens['e1s']} {e1_trigger} "
        e1e_anchor_offset = len(e1_anchor_temp)
        e1_anchor_temp += f"{s_tokens['e1e']} expresses a {s_tokens['mask']} event"
        e2_anchor_temp = "Here "
        e2s_anchor_offset = len(e2_anchor_temp)
        e2_anchor_temp += f"{s_tokens['e2s']} {e2_trigger} "
        e2e_anchor_offset = len(e2_anchor_temp)
        e2_anchor_temp += f"{s_tokens['e2e']} expresses a {s_tokens['mask']} event"
    elif anchor_temp_type.startswith("s"):  # soft template
        e1_anchor_temp = f"{s_tokens['l1']} {s_tokens['mask']} {s_tokens['l5']} "
        e1s_anchor_offset = len(e1_anchor_temp)
        e1_anchor_temp += f"{s_tokens['e1s']} {e1_trigger} "
        e1e_anchor_offset = len(e1_anchor_temp)
        e1_anchor_temp += f"{s_tokens['e1e']} {s_tokens['l2']}"
        e2_anchor_temp = f"{s_tokens['l3']} {s_tokens['mask']} {s_tokens['l6']} "
        e2s_anchor_offset = len(e2_anchor_temp)
        e2_anchor_temp += f"{s_tokens['e2s']} {e2_trigger} "
        e2e_anchor_offset = len(e2_anchor_temp)
        e2_anchor_temp += f"{s_tokens['e2e']} {s_tokens['l4']}"
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    if anchor_temp_type.endswith("tao"):
        e1_anchor_temp += f"{' ' + e1_arg_str if e1_arg_str else ''}{' ' + e1_related_str if e1_related_str else ''}."
        e2_anchor_temp += f"{' ' + e2_arg_str if e2_arg_str else ''}{' ' + e2_related_str if e2_related_str else ''}."
    elif anchor_temp_type.endswith("ta"):
        e1_anchor_temp += f"{' ' + e1_arg_str if e1_arg_str else ''}."
        e2_anchor_temp += f"{' ' + e2_arg_str if e2_arg_str else ''}."
    elif anchor_temp_type.endswith("t"):
        e1_anchor_temp += f"."
        e2_anchor_temp += f"."
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    # inference template
    infer_trigger_offsets = []
    infer_template = f"In conclusion, the events expressed by {s_tokens['e1s']} "
    infer_trigger_offsets.append([len(infer_template), len(infer_template) + len(e1_trigger) - 1])
    infer_template += f"{e1_trigger} {s_tokens['e1e']} and {s_tokens['e2s']} "
    infer_trigger_offsets.append([len(infer_template), len(infer_template) + len(e2_trigger) - 1])
    infer_template += f"{e2_trigger} {s_tokens['e2e']}"
    if remove_match or remove_subtype_match or remove_arg_match:
        if remove_match:
            infer_template += f" refer to {s_tokens['mask']} event."
        elif remove_subtype_match:
            infer_template += f" have {s_tokens['mask']} participants, so they refer to {s_tokens['mask']} event."
        elif remove_arg_match:
            infer_template += f" have {s_tokens['mask']} event type, so they refer to {s_tokens['mask']} event."
    else:
        infer_template += f" have {s_tokens['mask']} event type and {s_tokens['mask']} participants"
        if inference_temp_type == "hn":
            infer_template += f", so they refer to {s_tokens['mask']} event."
        elif inference_temp_type == "hc":
            infer_template += f". So the event expressed by {s_tokens['e1s']} "
            infer_trigger_offsets.append([len(infer_template), len(infer_template) + len(e1_trigger) - 1])
            infer_template += (
                f"{e1_trigger} {s_tokens['e1e']} {s_tokens['mask']} the event expressed by {s_tokens['e2s']} "
            )
            infer_trigger_offsets.append([len(infer_template), len(infer_template) + len(e2_trigger) - 1])
            infer_template += f"{e2_trigger} {s_tokens['e2e']}."
        elif inference_temp_type == "hq":
            infer_template += f". So do they refer to the same event? {s_tokens['mask']}."
        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
    special_tokens = (
        [s_tokens["e1s"], s_tokens["e1e"], s_tokens["e2s"], s_tokens["e2e"]]
        if anchor_temp_type.startswith("h")
        else [
            s_tokens["e1s"],
            s_tokens["e1e"],
            s_tokens["e2s"],
            s_tokens["e2e"],
            s_tokens["l1"],
            s_tokens["l2"],
            s_tokens["l3"],
            s_tokens["l4"],
            s_tokens["l5"],
            s_tokens["l6"],
        ]
    )
    if not remove_anchor_temp:
        special_tokens += [s_tokens[f"st{i}"] for i in range(len(EVENT_SUBTYPES) + 1)]
    if not remove_match:
        special_tokens += [s_tokens["match"], s_tokens["mismatch"]]
    if "c" in inference_temp_type:  # connect template
        special_tokens += [s_tokens["refer"], s_tokens["no_refer"]]
    return {
        "prefix_template": "" if remove_prefix_temp else prefix_template,
        "e1_anchor_template": "" if remove_anchor_temp else e1_anchor_temp,
        "e2_anchor_template": "" if remove_anchor_temp else e2_anchor_temp,
        "infer_template": infer_template,
        "prefix_trigger_offsets": [] if remove_prefix_temp else prefix_trigger_offsets,
        "infer_trigger_offsets": infer_trigger_offsets,
        "e1s_anchor_offset": -1 if remove_anchor_temp else e1s_anchor_offset,
        "e1e_anchor_offset": -1 if remove_anchor_temp else e1e_anchor_offset,
        "e2s_anchor_offset": -1 if remove_anchor_temp else e2s_anchor_offset,
        "e2e_anchor_offset": -1 if remove_anchor_temp else e2e_anchor_offset,
        "special_tokens": special_tokens,
    }


def create_arg_and_related_info_str(
    prompt_type: str, e1_related_info: dict, e2_related_info: dict, select_arg_strategy: str, s_tokens: dict
):
    assert select_arg_strategy in ["no_filter", "filter_related_args", "filter_all"]

    def select_args(my_args: list, other_related_info: dict, match_other_related_args: bool) -> list:
        if not my_args:
            return []
        other_has_part, other_has_place = False, False
        if match_other_related_args:
            other_args = other_related_info["arguments"] + list(
                filter(lambda x: x["mention"].lower() not in WORD_FILTER, other_related_info["related_arguments"])
            )
        else:
            other_args = other_related_info["arguments"]
        for arg in other_args:
            if arg["role"] == "participant":
                other_has_part = True
            if arg["role"] == "place":
                other_has_place = True
        return [
            arg
            for arg in my_args
            if (arg["role"] == "participant" and other_has_part) or (arg["role"] == "place" and other_has_place)
        ]

    def convert_args_to_str(args: list, use_filter: bool, soft_prompt: bool):
        if use_filter:
            args = filter(lambda x: x["mention"].lower() not in WORD_FILTER, args)
        if soft_prompt:
            return (
                f"{s_tokens['l7']} {', '.join([arg['mention'] for arg in args])} {s_tokens['l8']}".strip()
                if args
                else "",
                [s_tokens["l7"], s_tokens["l8"]],
            )
        participants, places, unknows = (
            [arg for arg in args if arg["role"] == "participant"],
            [arg for arg in args if arg["role"] == "place"],
            [arg for arg in args if arg["role"] == "unk"],
        )
        arg_str = ""
        if participants:
            participants.sort(key=lambda x: x["global_offset"])
            arg_str = f"with {', '.join([arg['mention'] for arg in participants])} as participants"
        if places:
            places.sort(key=lambda x: x["global_offset"])
            arg_str += f" at {', '.join([arg['mention'] for arg in places])}"
        if unknows:
            arg_str += f" (other arguments are {', '.join([arg['mention'] for arg in unknows])})"
        return arg_str.strip(), []

    def convert_related_info_to_str(related_triggers: list, related_args: list, use_filter: bool, soft_prompt: bool):
        if use_filter:
            related_args = list(filter(lambda x: x["mention"].lower() not in WORD_FILTER, related_args))
        if soft_prompt:
            return (
                f"{', '.join(set(related_triggers))} {s_tokens['l9']} "
                if related_triggers
                else "" f"{', '.join([arg['mention'] for arg in related_args])} {s_tokens['l10']}"
                if related_args
                else ""
            ).strip(), [s_tokens["l9"], s_tokens["l10"]]
        related_str = ""
        if related_triggers:
            related_str = f"(with related events: {', '.join(set(related_triggers))}"
            related_str += (
                f", and related participants/places: {', '.join([arg['mention'] for arg in related_args])})"
                if related_args
                else ")"
            )
        elif related_args:
            related_str = f"(with related participants/places: {', '.join([arg['mention'] for arg in related_args])})"
        return related_str.strip(), []

    special_tokens = []
    e1_args = (
        select_args(e1_related_info["arguments"], e2_related_info, "tao" in prompt_type)
        if select_arg_strategy == "filter_all"
        else e1_related_info["arguments"]
    )
    e2_args = (
        select_args(e2_related_info["arguments"], e1_related_info, "tao" in prompt_type)
        if select_arg_strategy == "filter_all"
        else e2_related_info["arguments"]
    )
    e1_arg_str, arg_special_tokens = convert_args_to_str(e1_args, not prompt_type.startswith("m"), "st" in prompt_type)
    e2_arg_str, _ = convert_args_to_str(e2_args, not prompt_type.startswith("m"), "st" in prompt_type)
    special_tokens += arg_special_tokens
    e1_related_triggers, e2_related_triggers = e1_related_info["related_triggers"], e2_related_info["related_triggers"]
    if not e1_related_triggers or not e2_related_triggers:
        e1_related_triggers, e2_related_triggers = [], []
    e1_related_args = (
        select_args(e1_related_info["related_arguments"], e2_related_info, "tao" in prompt_type)
        if select_arg_strategy in ["filter_all", "filter_related_args"]
        else e1_related_info["related_arguments"]
    )
    e2_related_args = (
        select_args(e2_related_info["related_arguments"], e1_related_info, "tao" in prompt_type)
        if select_arg_strategy in ["filter_all", "filter_related_args"]
        else e2_related_info["related_arguments"]
    )
    e1_related_str, related_special_tokens = convert_related_info_to_str(
        e1_related_triggers, e1_related_args, True, "st" in prompt_type
    )
    e2_related_str, _ = convert_related_info_to_str(e2_related_triggers, e2_related_args, True, "st" in prompt_type)
    special_tokens += related_special_tokens
    return e1_arg_str, e2_arg_str, e1_related_str, e2_related_str, list(set(special_tokens))


def findall(p, s):
    """yields all the positions of p in s."""
    i = s.find(p)
    while i != -1:
        yield i
        i = s.find(p, i + 1)


def get_event_cluster_size_ecr_data(event_id: str, events: list) -> int:
    for event in events:
        if event_id == event.event_id:
            return len(event.mentions)
    raise ValueError(f"Unknown event_id: {event_id}")


def get_event_cluster_size_no_ecr_data(event_id: str, clusters: list) -> int:
    for cluster in clusters:
        if event_id in cluster["events"]:
            return len(cluster["events"])
    raise ValueError(f"Unknown event_id: {event_id}")


def create_prompt(
    e1_sent_idx: int,
    e1_sent_start: int,
    e1_trigger: str,
    e1_related_info: dict,
    e2_sent_idx: int,
    e2_sent_start: int,
    e2_trigger: str,
    e2_related_info: dict,
    sentences: list,
    sentence_lens: list,
    prompt_type: str,
    select_arg_strategy: str,
    tokenizer,
    max_length: int,
) -> dict:
    """create event coreference prompts
    [e1/e2]_sent_idx:
        host sentence index
    [e1/e2]_sent_start:
        trigger offset in the host sentence
    [e1/e2]_trigger:
        trigger of the event
    [e1/e2]_related_info:
        arguments & related event information dict
    sentences:
        all the sentences in the document, {"start": sentence offset in the document, "text": content}
    sentence_lens:
        token numbers of all the sentences (not include [CLS], [SEP], etc.)
    prompt_type:
        prompt type
    select_arg_strategy:
        argument select strategy
    tokenizer:
        tokenizer of the chosen PTM
    max_length:
        max total token numbers of prompt
    # Return
    {
        'prompt': prompt, \n
        'mask_offset': coreference mask offset in the prompt, \n
        'type_match_mask_offset': event type match mask offset in the prompt, \n
        'arg_match_mask_offset': argument match mask offset in the prompt, \n
        'e1s_offset': [e1s] offset in the prompt, \n
        'e1e_offset': [e1e] offset in the prompt, \n
        'e1_type_mask_offset': e1 event type mask offset in the prompt, \n
        'e2s_offset': [e2s] offset in the prompt, \n
        'e2e_offset': [e2e] offset in the prompt, \n
        'e2_type_mask_offset': e2 event type mask offset in the prompt, \n
        'trigger_offsets': all the triggers' offsets in the prompt
    }
    """

    special_token_dict = {
        "mask": "<mask>",
        "e1s": "<e1_start>",
        "e1e": "<e1_end>",
        "e2s": "<e2_start>",
        "e2e": "<e2_end>",
        "l1": "<l1>",
        "l2": "<l2>",
        "l3": "<l3>",
        "l4": "<l4>",
        "l5": "<l5>",
        "l6": "<l6>",
        "l7": "<l7>",
        "l8": "<l8>",
        "l9": "<l9>",
        "l10": "<l10>",
        "match": "<match>",
        "mismatch": "<mismatch>",
        "refer": "<refer_to>",
        "no_refer": "<not_refer_to>",
    }
    for i in range(len(EVENT_SUBTYPES) + 1):
        special_token_dict[f"st{i}"] = f"<st_{i}>"

    if prompt_type.startswith("h") or prompt_type.startswith("s"):  # base prompt
        template_data = create_base_template(e1_trigger, e2_trigger, prompt_type, special_token_dict)
        trigger_offsets = template_data["trigger_offsets"]
        assert set(template_data["special_tokens"]).issubset(set(tokenizer.additional_special_tokens))
        template_length = len(tokenizer.tokenize(template_data["template"])) + 3
        context_data = create_event_context(
            e1_sent_idx,
            e1_sent_start,
            e1_trigger,
            e2_sent_idx,
            e2_sent_start,
            e2_trigger,
            sentences,
            sentence_lens,
            special_token_dict,
            tokenizer,
            max_length - template_length,
        )
        e1s_offset, e1e_offset = context_data["e1s_core_offset"], context_data["e1e_core_offset"]
        e2s_offset, e2e_offset = context_data["e2s_core_offset"], context_data["e2e_core_offset"]
        if context_data["type"] == "same_sent":  # two events in the same sentence
            prompt = (
                template_data["template"]
                + context_data["before_context"]
                + context_data["core_context"]
                + context_data["after_context"]
            )
            e1s_offset, e1e_offset, e2s_offset, e2e_offset = np.asarray(
                [e1s_offset, e1e_offset, e2s_offset, e2e_offset]
            ) + np.full((4,), len(template_data["template"] + context_data["before_context"]))
        else:  # two events in different sentences
            prompt = (
                template_data["template"]
                + context_data["e1_before_context"]
                + context_data["e1_core_context"]
                + context_data["e1_after_context"]
                + " "
                + context_data["e2_before_context"]
                + context_data["e2_core_context"]
                + context_data["e2_after_context"]
            )
            e1s_offset, e1e_offset = np.asarray([e1s_offset, e1e_offset]) + np.full(
                (2,), len(template_data["template"] + context_data["e1_before_context"])
            )
            e2s_offset, e2e_offset = np.asarray([e2s_offset, e2e_offset]) + np.full(
                (2,),
                len(template_data["template"])
                + len(context_data["e1_before_context"])
                + len(context_data["e1_core_context"])
                + len(context_data["e1_after_context"])
                + 1
                + len(context_data["e2_before_context"]),
            )
        mask_offset = prompt.find(special_token_dict["mask"])
        trigger_offsets.append([e1s_offset + len(special_token_dict["e1s"]) + 1, e1e_offset - 2])
        trigger_offsets.append([e2s_offset + len(special_token_dict["e2s"]) + 1, e2e_offset - 2])
        assert prompt[mask_offset : mask_offset + len(special_token_dict["mask"])] == special_token_dict["mask"]
        assert prompt[e1s_offset:e1e_offset] == special_token_dict["e1s"] + " " + e1_trigger + " "
        assert prompt[e1e_offset : e1e_offset + len(special_token_dict["e1e"])] == special_token_dict["e1e"]
        assert prompt[e2s_offset:e2e_offset] == special_token_dict["e2s"] + " " + e2_trigger + " "
        assert prompt[e2e_offset : e2e_offset + len(special_token_dict["e2e"])] == special_token_dict["e2e"]
        for s, e in trigger_offsets:
            assert prompt[s : e + 1] in [e1_trigger, e2_trigger]
        return {
            "prompt": prompt,
            "mask_offset": mask_offset,
            "type_match_mask_offset": -1,
            "arg_match_mask_offset": -1,
            "e1s_offset": e1s_offset,
            "e1e_offset": e1e_offset,
            "e1_type_mask_offset": -1,
            "e2s_offset": e2s_offset,
            "e2e_offset": e2e_offset,
            "e2_type_mask_offset": -1,
            "trigger_offsets": trigger_offsets,
        }
    elif prompt_type.startswith("m"):  # mix prompt
        remove_anchor_temp, remove_match, remove_subtype_match, remove_arg_match = (
            (prompt_type == "ma_remove-anchor"),
            (prompt_type == "ma_remove-match"),
            (prompt_type == "ma_remove-subtype-match"),
            (prompt_type == "ma_remove-arg-match"),
        )
        e1_arg_str, e2_arg_str, e1_related_str, e2_related_str, special_tokens = create_arg_and_related_info_str(
            prompt_type, e1_related_info, e2_related_info, select_arg_strategy, special_token_dict
        )
        template_data = create_mix_template(
            e1_trigger,
            e2_trigger,
            e1_arg_str,
            e2_arg_str,
            e1_related_str,
            e2_related_str,
            prompt_type,
            special_token_dict,
        )
        template_length = (
            len(tokenizer.tokenize(template_data["prefix_template"]))
            + len(tokenizer.tokenize(template_data["e1_anchor_template"]))
            + len(tokenizer.tokenize(template_data["e2_anchor_template"]))
            + len(tokenizer.tokenize(template_data["infer_template"]))
            + 6
        )
        trigger_offsets = template_data["prefix_trigger_offsets"]
        assert set(([] if remove_anchor_temp else special_tokens) + template_data["special_tokens"]).issubset(
            set(tokenizer.additional_special_tokens)
        )
        context_data = create_event_context(
            e1_sent_idx,
            e1_sent_start,
            e1_trigger,
            e2_sent_idx,
            e2_sent_start,
            e2_trigger,
            sentences,
            sentence_lens,
            special_token_dict,
            tokenizer,
            max_length - template_length,
        )
        e1s_offset, e1e_offset = template_data["e1s_anchor_offset"], template_data["e1e_anchor_offset"]
        e2s_offset, e2e_offset = template_data["e2s_anchor_offset"], template_data["e2e_anchor_offset"]
        e1s_context_offset, e1e_context_offset = context_data["e1s_core_offset"], context_data["e1e_core_offset"]
        e2s_context_offset, e2e_context_offset = context_data["e2s_core_offset"], context_data["e2e_core_offset"]
        infer_trigger_offsets = template_data["infer_trigger_offsets"]
        if context_data["type"] == "same_sent":  # two events in the same sentence
            prompt = (
                template_data["prefix_template"] + context_data["before_context"] + context_data["core_context"] + " "
            )
            e1s_offset, e1e_offset = np.asarray([e1s_offset, e1e_offset]) + np.full((2,), len(prompt))
            prompt += template_data["e1_anchor_template"] + " "
            e2s_offset, e2e_offset = np.asarray([e2s_offset, e2e_offset]) + np.full((2,), len(prompt))
            prompt += (
                template_data["e2_anchor_template"]
                + context_data["after_context"]
                + " "
                + template_data["infer_template"]
            )
            e1s_context_offset, e1e_context_offset, e2s_context_offset, e2e_context_offset = np.asarray(
                [e1s_context_offset, e1e_context_offset, e2s_context_offset, e2e_context_offset]
            ) + np.full((4,), len(template_data["prefix_template"]) + len(context_data["before_context"]))
            if remove_anchor_temp:
                e1s_offset, e1e_offset, e2s_offset, e2e_offset = (
                    e1s_context_offset,
                    e1e_context_offset,
                    e2s_context_offset,
                    e2e_context_offset,
                )
            infer_temp_offset = (
                len(template_data["prefix_template"])
                + len(context_data["before_context"])
                + len(context_data["core_context"])
                + 1
                + len(template_data["e1_anchor_template"])
                + 1
                + len(template_data["e2_anchor_template"])
                + len(context_data["after_context"])
                + 1
            )
            infer_trigger_offsets = [[s + infer_temp_offset, e + infer_temp_offset] for s, e in infer_trigger_offsets]
        else:  # two events in different sentences
            prompt = (
                template_data["prefix_template"]
                + context_data["e1_before_context"]
                + context_data["e1_core_context"]
                + " "
            )
            e1s_offset, e1e_offset = np.asarray([e1s_offset, e1e_offset]) + np.full((2,), len(prompt))
            prompt += (
                template_data["e1_anchor_template"]
                + context_data["e1_after_context"]
                + " "
                + context_data["e2_before_context"]
                + context_data["e2_core_context"]
                + " "
            )
            e2s_offset, e2e_offset = np.asarray([e2s_offset, e2e_offset]) + np.full((2,), len(prompt))
            prompt += (
                template_data["e2_anchor_template"]
                + context_data["e2_after_context"]
                + " "
                + template_data["infer_template"]
            )
            e1s_context_offset, e1e_context_offset = np.asarray([e1s_context_offset, e1e_context_offset]) + np.full(
                (2,), len(template_data["prefix_template"] + context_data["e1_before_context"])
            )
            e2s_context_offset, e2e_context_offset = np.asarray([e2s_context_offset, e2e_context_offset]) + np.full(
                (2,),
                len(template_data["prefix_template"])
                + len(context_data["e1_before_context"])
                + len(context_data["e1_core_context"])
                + 1
                + len(template_data["e1_anchor_template"])
                + len(context_data["e1_after_context"])
                + 1
                + len(context_data["e2_before_context"]),
            )
            if remove_anchor_temp:
                e1s_offset, e1e_offset, e2s_offset, e2e_offset = (
                    e1s_context_offset,
                    e1e_context_offset,
                    e2s_context_offset,
                    e2e_context_offset,
                )
            infer_temp_offset = (
                len(template_data["prefix_template"])
                + len(context_data["e1_before_context"])
                + len(context_data["e1_core_context"])
                + 1
                + len(template_data["e1_anchor_template"])
                + len(context_data["e1_after_context"])
                + 1
                + len(context_data["e2_before_context"])
                + len(context_data["e2_core_context"])
                + 1
                + len(template_data["e2_anchor_template"])
                + len(context_data["e2_after_context"])
                + 1
            )
            infer_trigger_offsets = [[s + infer_temp_offset, e + infer_temp_offset] for s, e in infer_trigger_offsets]
        mask_offsets = list(findall(special_token_dict["mask"], prompt))
        assert len(mask_offsets) == (
            3 if remove_anchor_temp or remove_match else 4 if remove_subtype_match or remove_arg_match else 5
        )
        if remove_anchor_temp:
            type_match_mask_offset, arg_match_mask_offset, mask_offset = mask_offsets
        else:
            if remove_match:
                e1_type_mask_offset, e2_type_mask_offset, mask_offset = mask_offsets
            elif remove_subtype_match:
                e1_type_mask_offset, e2_type_mask_offset, arg_match_mask_offset, mask_offset = mask_offsets
            elif remove_arg_match:
                e1_type_mask_offset, e2_type_mask_offset, type_match_mask_offset, mask_offset = mask_offsets
            else:
                (
                    e1_type_mask_offset,
                    e2_type_mask_offset,
                    type_match_mask_offset,
                    arg_match_mask_offset,
                    mask_offset,
                ) = mask_offsets
            assert (
                prompt[e1_type_mask_offset : e1_type_mask_offset + len(special_token_dict["mask"])]
                == special_token_dict["mask"]
            )
            assert (
                prompt[e2_type_mask_offset : e2_type_mask_offset + len(special_token_dict["mask"])]
                == special_token_dict["mask"]
            )
        trigger_offsets.append([e1s_context_offset + len(special_token_dict["e1s"]) + 1, e1e_context_offset - 2])
        trigger_offsets.append([e2s_context_offset + len(special_token_dict["e2s"]) + 1, e2e_context_offset - 2])
        if not remove_anchor_temp:
            trigger_offsets.append([e1s_offset + len(special_token_dict["e1s"]) + 1, e1e_offset - 2])
            trigger_offsets.append([e2s_offset + len(special_token_dict["e2s"]) + 1, e2e_offset - 2])
        trigger_offsets += infer_trigger_offsets
        if not remove_match:
            if not remove_subtype_match:
                assert (
                    prompt[type_match_mask_offset : type_match_mask_offset + len(special_token_dict["mask"])]
                    == special_token_dict["mask"]
                )
            if not remove_arg_match:
                assert (
                    prompt[arg_match_mask_offset : arg_match_mask_offset + len(special_token_dict["mask"])]
                    == special_token_dict["mask"]
                )
        assert prompt[mask_offset : mask_offset + len(special_token_dict["mask"])] == special_token_dict["mask"]
        assert prompt[e1s_offset:e1e_offset] == special_token_dict["e1s"] + " " + e1_trigger + " "
        assert prompt[e1e_offset : e1e_offset + len(special_token_dict["e1e"])] == special_token_dict["e1e"]
        assert prompt[e2s_offset:e2e_offset] == special_token_dict["e2s"] + " " + e2_trigger + " "
        assert prompt[e2e_offset : e2e_offset + len(special_token_dict["e2e"])] == special_token_dict["e2e"]
        for s, e in trigger_offsets:
            assert prompt[s : e + 1] == e1_trigger or prompt[s : e + 1] == e2_trigger
        return {
            "prompt": prompt,
            "mask_offset": mask_offset,
            "type_match_mask_offset": -1 if remove_match or remove_subtype_match else type_match_mask_offset,
            "arg_match_mask_offset": -1 if remove_match or remove_arg_match else arg_match_mask_offset,
            "e1s_offset": e1s_offset,
            "e1e_offset": e1e_offset,
            "e1_type_mask_offset": -1 if remove_anchor_temp else e1_type_mask_offset,
            "e2s_offset": e2s_offset,
            "e2e_offset": e2e_offset,
            "e2_type_mask_offset": -1 if remove_anchor_temp else e2_type_mask_offset,
            "trigger_offsets": trigger_offsets,
        }


def create_event_simi_dict(event_pairs_id: list, event_pairs_cos: list, clusters: list) -> dict:
    """create similar event list for each event
    # Args
    event_pairs_id:
        event-pair id, format: 'e1_id###e2_id'
    event_pairs_cos:
        similarities of event pairs
    clusters:
        event clusters in the document
    # Return
    {
        event id: [{'id': event id, 'cos': event similarity, 'coref': coreference}, ...]
    }
    """
    simi_dict = defaultdict(list)
    for id_pair, cos in zip(event_pairs_id, event_pairs_cos):
        e1_id, e2_id = id_pair.split("###")
        coref = 1 if get_event_cluster_id(e1_id, clusters) == get_event_cluster_id(e2_id, clusters) else 0
        simi_dict[e1_id].append({"id": e2_id, "cos": cos, "coref": coref})
        simi_dict[e2_id].append({"id": e1_id, "cos": cos, "coref": coref})
    for simi_list in simi_dict.values():
        simi_list.sort(key=lambda x: x["cos"], reverse=True)
    return simi_dict


def get_noncoref_ids(simi_list: list, top_k: int) -> list:
    """get non-coreference event list
    # Args
    simi_list:
        similar event list, format: [{'id': event id, 'cos': event similarity, 'coref': coreference}, ...]
    top_k:
        maximum return event number
    # Return
    non-coreference event id list
    """
    noncoref_ids = []
    for simi in simi_list:
        if simi["coref"] == 0:
            noncoref_ids.append(simi["id"])
            if len(noncoref_ids) >= top_k:
                break
    return noncoref_ids


def get_event_pair_similarity(simi_dict: dict, e1_id: str, e2_id: str) -> float:
    for item in simi_dict[e1_id]:
        if item["id"] == e2_id:
            return item["cos"]
    raise ValueError(f"Can't find event pair: {e1_id} & {e2_id}")


def get_event_cluster_id(event_id: str, clusters: list) -> str:
    for cluster in clusters:
        if event_id in cluster["events"]:
            return cluster["hopper_id"]
    raise ValueError(f"Unknown event_id: {event_id}")


class CorefPromptCoreferenceDataset(Dataset):
    def __init__(
        self,
        ecr_data: EcrData,
        tokenizer: RobertaTokenizer,
        simi_file: str,
        prompt_type: str,
        select_arg_strategy: str,
        max_length: int,
        mode: str,
    ):
        self.ecr_data = ecr_data
        self.simi_file = simi_file
        self.prompt_type = prompt_type
        self.select_arg_strategy = select_arg_strategy
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.documents = self.ecr_data.documents
        self.events = self.ecr_data.events
        self.mentions = self.ecr_data.mentions

        self.related_dict = get_pred_related_info(self.simi_file)

        self.data = self.adapt_data()
        if mode == "train":
            self.data = self.data_sampling()
        else:
            self.data = self.data

    def data_sampling(self):
        pos_data, neg_data = [], []

        for d in self.data:
            if d["label"] == 1:
                pos_data.append(d)
            else:
                neg_data.append(d)
        nl = len(neg_data)
        neg_data = random.sample(neg_data, int(nl * 0.1))
        data = neg_data + pos_data
        return data

    def adapt_data(self):
        data = []
        doc2mentions = defaultdict(list)
        for _, mention in self.mentions.items():
            doc_id = mention.doc_id
            doc2mentions[doc_id].append(mention)

        for doc_id, doc in self.documents.items():
            doc_mentions = doc2mentions[doc_id]
            sentences = doc.meta["sentences"]
            sentences_lengths = [len(self.tokenizer.tokenize(sent["text"])) for sent in sentences]
            # create event pairs
            for i in range(len(doc_mentions) - 1):
                for j in range(i + 1, len(doc_mentions)):
                    event_1, event_2 = doc_mentions[i], doc_mentions[j]
                    event_1_cluster_id, event_2_cluster_id = event_1.meta["event_id"], event_2.meta["event_id"]
                    event_1_related_info = self.related_dict[doc_id][event_1.extent.start + event_1.anchor.start]
                    event_2_related_info = self.related_dict[doc_id][event_2.extent.start + event_2.anchor.start]
                    prompt_data = create_prompt(
                        event_1.meta["sent_idx"],
                        event_1.anchor.start,
                        event_1.anchor.text,
                        event_1_related_info,
                        event_2.meta["sent_idx"],
                        event_2.anchor.start,
                        event_2.anchor.text,
                        event_2_related_info,
                        sentences,
                        sentences_lengths,
                        self.prompt_type,
                        self.select_arg_strategy,
                        self.tokenizer,
                        self.max_length,
                    )
                    data.append(
                        {
                            "id": doc_id,
                            "prompt": prompt_data["prompt"],
                            "mask_offset": prompt_data["mask_offset"],
                            "type_match_mask_offset": prompt_data["type_match_mask_offset"],
                            "arg_match_mask_offset": prompt_data["arg_match_mask_offset"],
                            "trigger_offsets": prompt_data["trigger_offsets"],
                            "e1_id": event_1.extent.start + event_1.anchor.start,  # event1
                            "e1_trigger": event_1.anchor.text,
                            "e1_subtype": event_1.meta["mention_subtype"]
                            if event_1.meta["mention_subtype"] in EVENT_SUBTYPES
                            else "normal",
                            "e1_subtype_id": subtype2id.get(event_1.meta["mention_subtype"], 0),  # 0 - 'other'
                            "e1_coref_link_len": get_event_cluster_size_ecr_data(event_1.meta["event_id"], self.events),
                            "e1s_offset": prompt_data["e1s_offset"],
                            "e1e_offset": prompt_data["e1e_offset"],
                            "e1_type_mask_offset": prompt_data["e1_type_mask_offset"],
                            "e2_id": event_2.extent.start + event_2.anchor.start,  # event2
                            "e2_trigger": event_2.anchor.text,
                            "e2_subtype": event_2.meta["mention_subtype"]
                            if event_1.meta["mention_subtype"] in EVENT_SUBTYPES
                            else "normal",
                            "e2_subtype_id": subtype2id.get(event_2.meta["mention_subtype"], 0),  # 0 - 'other'
                            "e2_coref_link_len": get_event_cluster_size_ecr_data(event_2.meta["event_id"], self.events),
                            "e2s_offset": prompt_data["e2s_offset"],
                            "e2e_offset": prompt_data["e2e_offset"],
                            "e2_type_mask_offset": prompt_data["e2_type_mask_offset"],
                            "label": 1 if event_1_cluster_id == event_2_cluster_id else 0,
                            "mention1_id": event_1.mention_id,
                            "mention2_id": event_2.mention_id,
                        }
                    )
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# class CorefPromptCoreferenceTinyDataset(Dataset):
#     def __init__(
#         self,
#         ecr_data: EcrData,
#         tokenizer: RobertaTokenizer,
#         data_file_with_cos: str,
#         simi_file: str,
#         prompt_type: str,
#         select_arg_strategy: str,
#         model_type: str,
#         max_length: int,
#         sample_strategy: str,
#         neg_top_k: int,
#         neg_threshold: float,
#     ):
#         assert (
#             prompt_type in PROMPT_TYPE
#             and select_arg_strategy in SELECT_ARG_STRATEGY
#             and model_type in ["bert", "roberta"]
#         )
#         assert sample_strategy in ["random", "corefnm", "corefenn1", "corefenn2"]
#         assert neg_top_k > 0
#         self.is_easy_to_judge = lambda simi_list, top_k: len(set([simi["coref"] for simi in simi_list[:top_k]])) == 1

#         self.model_type = model_type
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         self.sample_strategy = sample_strategy
#         self.related_dict = get_pred_related_info(simi_file)
#         # EcrData
#         self.ecr_data = ecr_data
#         self.documents = self.ecr_data.documents
#         self.events = self.ecr_data.events
#         self.mentions = self.ecr_data.mentions
#         self.data = self.adapt_data(data_file_with_cos, neg_top_k, neg_threshold, prompt_type, select_arg_strategy)

#     def adapt_data(self, data_file_with_cos, neg_top_k, neg_threshold, prompt_type, select_arg_strategy):
#         data = []
#         doc2mentions = defaultdict(list)
#         for _, mention in self.mentions.items():
#             doc_id = mention.doc_id
#             doc2mentions[doc_id].append(mention)
#         with open(data_file_with_cos, "rt", encoding="utf-8") as f_cos:
#             for doc_id, doc in self.documents.items():
#                 doc_mentions = doc2mentions[doc_id]
#                 sentences = doc.meta["sentences"]
#                 sentences_lengths = [len(self.tokenizer.tokenize(sent["text"])) for sent in sentences]
#                 # create event pairs
#                 for i in range(len(doc_mentions) - 1):
#                     for j in range(i + 1, len(doc_mentions)):
#                         event_1, event_2 = doc_mentions[i], doc_mentions[j]
#                         event_1_cluster_id, event_2_cluster_id = event_1.meta["event_id"], event_2.meta["event_id"]
#                         event_1_related_info = self.related_dict[doc_id][event_1.extent.start + event_1.anchor.start]
#                         event_2_related_info = self.related_dict[doc_id][event_2.extent.start + event_2.anchor.start]
#                         if event_1_cluster_id == event_2_cluster_id:
#                             prompt_data = create_prompt(
#                                 event_1.meta["sent_idx"],
#                                 event_1.anchor.start,
#                                 event_1.anchor.text,
#                                 event_1_related_info,
#                                 event_2.meta["sent_idx"],
#                                 event_2.anchor.start,
#                                 event_2.anchor.text,
#                                 event_2_related_info,
#                                 sentences,
#                                 sentences_lengths,
#                                 prompt_type,
#                                 select_arg_strategy,
#                                 self.model_type,
#                                 self.tokenizer,
#                                 self.max_length,
#                             )
#                             data.append(
#                                 {
#                                     "id": doc_id,
#                                     "prompt": prompt_data["prompt"],
#                                     "mask_offset": prompt_data["mask_offset"],
#                                     "type_match_mask_offset": prompt_data["type_match_mask_offset"],
#                                     "arg_match_mask_offset": prompt_data["arg_match_mask_offset"],
#                                     "trigger_offsets": prompt_data["trigger_offsets"],
#                                     "e1_id": event_1.extent.start + event_1.anchor.start,  # event1
#                                     "e1_trigger": event_1.anchor.text,
#                                     "e1_subtype": event_1.meta["mention_subtype"]
#                                     if event_1.meta["mention_subtype"] in EVENT_SUBTYPES
#                                     else "normal",
#                                     "e1_subtype_id": subtype2id.get(event_1.meta["mention_subtype"], 0),  # 0 - 'other'
#                                     "e1_coref_link_len": get_event_cluster_size_ecr_data(
#                                         event_1.meta["event_id"], self.events
#                                     ),
#                                     "e1s_offset": prompt_data["e1s_offset"],
#                                     "e1e_offset": prompt_data["e1e_offset"],
#                                     "e1_type_mask_offset": prompt_data["e1_type_mask_offset"],
#                                     "e2_id": event_2.extent.start + event_2.anchor.start,  # event2
#                                     "e2_trigger": event_2.anchor.text,
#                                     "e2_subtype": event_2.meta["mention_subtype"]
#                                     if event_1.meta["mention_subtype"] in EVENT_SUBTYPES
#                                     else "normal",
#                                     "e2_subtype_id": subtype2id.get(event_2.meta["mention_subtype"], 0),  # 0 - 'other'
#                                     "e2_coref_link_len": get_event_cluster_size_ecr_data(
#                                         event_2.meta["event_id"], self.events
#                                     ),
#                                     "e2s_offset": prompt_data["e2s_offset"],
#                                     "e2e_offset": prompt_data["e2e_offset"],
#                                     "e2_type_mask_offset": prompt_data["e2_type_mask_offset"],
#                                     "label": 1,
#                                     "mention1_id": event_1.mention_id,
#                                     "mention2_id": event_2.mention_id,
#                                 }
#                             )
#             # negtive samples (non-coref pairs)
#             if self.sample_strategy == "random":  # random undersampling
#                 doc_sent_dict, doc_sent_len_dict = {}, {}
#                 all_nocoref_event_pairs = []
#                 for doc_id, doc in self.documents.items():
#                     sentences = doc.meta["sentences"]
#                     doc_sent_dict[doc_id] = doc.meta["sentences"]
#                     doc_sent_len_dict[doc_id] = [len(self.tokenizer.tokenize(sent["text"])) for sent in sentences]
#                     doc_mentions = doc2mentions[doc_id]
#                     for i in range(len(doc_mentions) - 1):
#                         for j in range(i + 1, len(doc_mentions)):
#                             event_1, event_2 = doc_mentions[i], doc_mentions[j]
#                             event_1_cluster_id, event_2_cluster_id = event_1.meta["event_id"], event_2.meta["event_id"]
#                             event_1_cluster_size = get_event_cluster_size_ecr_data(
#                                 event_1.meta["event_id"], self.events
#                             )
#                             event_2_cluster_size = get_event_cluster_size_ecr_data(
#                                 event_2.meta["event_id"], self.events
#                             )
#                             event_1_related_info = self.related_dict[doc_id][
#                                 event_1.extent.start + event_1.anchor.start
#                             ]
#                             event_2_related_info = self.related_dict[doc_id][
#                                 event_2.extent.start + event_2.anchor.start
#                             ]
#                             if event_1_cluster_id != event_2_cluster_id:
#                                 all_nocoref_event_pairs.append(
#                                     (
#                                         doc_id,
#                                         event_1,
#                                         event_2,
#                                         event_1_related_info,
#                                         event_2_related_info,
#                                         event_1_cluster_size,
#                                         event_2_cluster_size,
#                                     )
#                                 )
#                 for choose_idx in np.random.choice(
#                     np.random.permutation(len(all_nocoref_event_pairs)), len(data), replace=False
#                 ):
#                     (
#                         doc_id,
#                         event_1,
#                         event_2,
#                         event_1_related_info,
#                         event_2_related_info,
#                         e1_cluster_size,
#                         e2_clister_size,
#                     ) = all_nocoref_event_pairs[choose_idx]
#                     prompt_data = create_prompt(
#                         event_1.meta["sent_idx"],
#                         event_1.anchor.start,
#                         event_1.anchor.text,
#                         event_1_related_info,
#                         event_2.meta["sent_idx"],
#                         event_2.anchor.start,
#                         event_2.anchor.text,
#                         event_2_related_info,
#                         sentences,
#                         sentences_lengths,
#                         prompt_type,
#                         select_arg_strategy,
#                         self.model_type,
#                         self.tokenizer,
#                         self.max_length,
#                     )
#                     data.append(
#                         {
#                             "id": doc_id,
#                             "prompt": prompt_data["prompt"],
#                             "mask_offset": prompt_data["mask_offset"],
#                             "type_match_mask_offset": prompt_data["type_match_mask_offset"],
#                             "arg_match_mask_offset": prompt_data["arg_match_mask_offset"],
#                             "trigger_offsets": prompt_data["trigger_offsets"],
#                             "e1_id": event_1.extent.start + event_1.anchor.start,  # event1
#                             "e1_trigger": event_1.anchor.text,
#                             "e1_subtype": event_1.meta["mention_subtype"]
#                             if event_1.meta["mention_subtype"] in EVENT_SUBTYPES
#                             else "normal",
#                             "e1_subtype_id": subtype2id.get(event_1.meta["mention_subtype"], 0),  # 0 - 'other'
#                             "e1_coref_link_len": e1_cluster_size,
#                             "e1s_offset": prompt_data["e1s_offset"],
#                             "e1e_offset": prompt_data["e1e_offset"],
#                             "e1_type_mask_offset": prompt_data["e1_type_mask_offset"],
#                             "e2_id": event_2.extent.start + event_2.anchor.start,  # event2
#                             "e2_trigger": event_2.anchor.text,
#                             "e2_subtype": event_2.meta["mention_subtype"]
#                             if event_1.meta["mention_subtype"] in EVENT_SUBTYPES
#                             else "normal",
#                             "e2_subtype_id": subtype2id.get(event_2.meta["mention_subtype"], 0),  # 0 - 'other'
#                             "e2_coref_link_len": e2_clister_size,
#                             "e2s_offset": prompt_data["e2s_offset"],
#                             "e2e_offset": prompt_data["e2e_offset"],
#                             "e2_type_mask_offset": prompt_data["e2_type_mask_offset"],
#                             "label": 0,
#                             "mention1_id": event_1.mention_id,
#                             "mention2_id": event_2.mention_id,
#                         }
#                     )
#             elif self.sample_strategy == "corefnm":  # CorefNearMiss
#                 for line in f_cos.readlines():
#                     sample = json.loads(line.strip())
#                     clusters = sample["clusters"]
#                     sentences = sample["sentences"]
#                     sentences_lengths = [len(self.tokenizer.tokenize(sent["text"])) for sent in sentences]
#                     event_simi_dict = create_event_simi_dict(
#                         sample["event_pairs_id"], sample["event_pairs_cos"], clusters
#                     )
#                     events_list, events_dict = sample["events"], {e["event_id"]: e for e in sample["events"]}
#                     for i in range(len(events_list)):
#                         event_1 = events_list[i]
#                         for e_id in get_noncoref_ids(
#                             event_simi_dict[event_1["event_id"]], top_k=neg_top_k
#                         ):  # non-coref
#                             event_2 = events_dict[e_id]
#                             if event_1["start"] < event_2["start"]:
#                                 event_1_related_info = self.related_dict[sample["doc_id"]][event_1["start"]]
#                                 event_2_related_info = self.related_dict[sample["doc_id"]][event_2["start"]]
#                                 prompt_data = create_prompt(
#                                     event_1["sent_idx"],
#                                     event_1["sent_start"],
#                                     event_1["trigger"],
#                                     event_1_related_info,
#                                     event_2["sent_idx"],
#                                     event_2["sent_start"],
#                                     event_2["trigger"],
#                                     event_2_related_info,
#                                     sentences,
#                                     sentences_lengths,
#                                     prompt_type,
#                                     select_arg_strategy,
#                                     self.model_type,
#                                     self.tokenizer,
#                                     self.max_length,
#                                 )
#                                 data.append(
#                                     {
#                                         "id": sample["doc_id"],
#                                         "prompt": prompt_data["prompt"],
#                                         "mask_offset": prompt_data["mask_offset"],
#                                         "type_match_mask_offset": prompt_data["type_match_mask_offset"],
#                                         "arg_match_mask_offset": prompt_data["arg_match_mask_offset"],
#                                         "trigger_offsets": prompt_data["trigger_offsets"],
#                                         "e1_id": event_1["start"],  # event1
#                                         "e1_trigger": event_1["trigger"],
#                                         "e1_subtype": event_1["subtype"]
#                                         if event_1["subtype"] in EVENT_SUBTYPES
#                                         else "normal",
#                                         "e1_subtype_id": subtype2id.get(event_1["subtype"], 0),  # 0 - 'other'
#                                         "e1_coref_link_len": get_event_cluster_size_no_ecr_data(
#                                             event_1["event_id"], clusters
#                                         ),
#                                         "e1s_offset": prompt_data["e1s_offset"],
#                                         "e1e_offset": prompt_data["e1e_offset"],
#                                         "e1_type_mask_offset": prompt_data["e1_type_mask_offset"],
#                                         "e2_id": event_2["start"],  # event2
#                                         "e2_trigger": event_2["trigger"],
#                                         "e2_subtype": event_2["subtype"]
#                                         if event_2["subtype"] in EVENT_SUBTYPES
#                                         else "normal",
#                                         "e2_subtype_id": subtype2id.get(event_2["subtype"], 0),  # 0 - 'other'
#                                         "e2_coref_link_len": get_event_cluster_size_no_ecr_data(
#                                             event_2["event_id"], clusters
#                                         ),
#                                         "e2s_offset": prompt_data["e2s_offset"],
#                                         "e2e_offset": prompt_data["e2e_offset"],
#                                         "e2_type_mask_offset": prompt_data["e2_type_mask_offset"],
#                                         "label": 0,
#                                     }
#                                 )
#             elif self.sample_strategy.startswith("corefenn"):  # Coref Edited Nearest Neighbours
#                 for line in f_cos.readlines():
#                     sample = json.loads(line.strip())
#                     clusters = sample["clusters"]
#                     sentences = sample["sentences"]
#                     sentences_lengths = [len(self.tokenizer.tokenize(sent["text"])) for sent in sentences]
#                     event_simi_dict = create_event_simi_dict(
#                         sample["event_pairs_id"], sample["event_pairs_cos"], clusters
#                     )
#                     events = sample["events"]
#                     for i in range(len(events) - 1):
#                         for j in range(i + 1, len(events)):
#                             event_1, event_2 = events[i], events[j]
#                             if self.sample_strategy == "corefenn1":
#                                 if self.is_easy_to_judge(  # e1 or e2 is easy to judge coreference
#                                     event_simi_dict[event_1["event_id"]], top_k=neg_top_k
#                                 ) or self.is_easy_to_judge(event_simi_dict[event_2["event_id"]], top_k=neg_top_k):
#                                     continue
#                             elif self.sample_strategy == "corefenn2":
#                                 if (
#                                     get_event_pair_similarity(event_simi_dict, event_1["event_id"], event_2["event_id"])
#                                     <= neg_threshold
#                                 ):
#                                     continue
#                             event_1_cluster_id = get_event_cluster_id(event_1["event_id"], clusters)
#                             event_2_cluster_id = get_event_cluster_id(event_2["event_id"], clusters)
#                             event_1_related_info = self.related_dict[sample["doc_id"]][event_1["start"]]
#                             event_2_related_info = self.related_dict[sample["doc_id"]][event_2["start"]]
#                             if event_1_cluster_id != event_2_cluster_id:
#                                 prompt_data = create_prompt(
#                                     event_1["sent_idx"],
#                                     event_1["sent_start"],
#                                     event_1["trigger"],
#                                     event_1_related_info,
#                                     event_2["sent_idx"],
#                                     event_2["sent_start"],
#                                     event_2["trigger"],
#                                     event_2_related_info,
#                                     sentences,
#                                     sentences_lengths,
#                                     prompt_type,
#                                     select_arg_strategy,
#                                     self.model_type,
#                                     self.tokenizer,
#                                     self.max_length,
#                                 )
#                                 data.append(
#                                     {
#                                         "id": sample["doc_id"],
#                                         "prompt": prompt_data["prompt"],
#                                         "mask_offset": prompt_data["mask_offset"],
#                                         "type_match_mask_offset": prompt_data["type_match_mask_offset"],
#                                         "arg_match_mask_offset": prompt_data["arg_match_mask_offset"],
#                                         "trigger_offsets": prompt_data["trigger_offsets"],
#                                         "e1_id": event_1["start"],  # event1
#                                         "e1_trigger": event_1["trigger"],
#                                         "e1_subtype": event_1["subtype"]
#                                         if event_1["subtype"] in EVENT_SUBTYPES
#                                         else "normal",
#                                         "e1_subtype_id": subtype2id.get(event_1["subtype"], 0),  # 0 - 'other'
#                                         "e1_coref_link_len": get_event_cluster_size_no_ecr_data(
#                                             event_1["event_id"], clusters
#                                         ),
#                                         "e1s_offset": prompt_data["e1s_offset"],
#                                         "e1e_offset": prompt_data["e1e_offset"],
#                                         "e1_type_mask_offset": prompt_data["e1_type_mask_offset"],
#                                         "e2_id": event_2["start"],  # event2
#                                         "e2_trigger": event_2["trigger"],
#                                         "e2_subtype": event_2["subtype"]
#                                         if event_2["subtype"] in EVENT_SUBTYPES
#                                         else "normal",
#                                         "e2_subtype_id": subtype2id.get(event_2["subtype"], 0),  # 0 - 'other'
#                                         "e2_coref_link_len": get_event_cluster_size_no_ecr_data(
#                                             event_2["event_id"], clusters
#                                         ),
#                                         "e2s_offset": prompt_data["e2s_offset"],
#                                         "e2e_offset": prompt_data["e2e_offset"],
#                                         "e2_type_mask_offset": prompt_data["e2_type_mask_offset"],
#                                         "label": 0,
#                                     }
#                                 )
#             else:
#                 raise ValueError(f"Unknown sampling type: {prompt_type}")
#         return data

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return self.data[idx]


class RobertaLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return x


class RobertaForBasePrompt(RobertaPreTrainedModel):
    def __init__(self, config, roberta, lm_head):
        super().__init__(config)
        self.roberta = roberta
        self.lm_head = lm_head
        #
        self.hidden_size = config.hidden_size

        self.span_extractor = SelfAttentiveSpanExtractor(input_dim=self.hidden_size)
        self.cosine_space_dim, self.cosine_slices, self.tensor_factor = (64, 128, 4)
        self.cosine_ffnn = nn.Linear(self.hidden_size, self.cosine_space_dim)
        self.cosine_mat_p = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_slices), requires_grad=True))
        self.cosine_mat_q = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_space_dim), requires_grad=True))
        self.mapping = nn.Linear(2 * self.hidden_size + self.cosine_slices, self.hidden_size)
        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])
        self.post_init()

    # multi_perspective_cosine
    def multi_perspective_cosine(self, cosine_ffnn, cosine_mat_p, cosine_mat_q, batch_event_1_reps, batch_event_2_reps):
        # batch_event_1
        batch_event_1_reps = cosine_ffnn(batch_event_1_reps)
        batch_event_1_reps = batch_event_1_reps.unsqueeze(dim=1)
        batch_event_1_reps = cosine_mat_q * batch_event_1_reps
        batch_event_1_reps = batch_event_1_reps.permute((0, 2, 1))
        batch_event_1_reps = torch.matmul(batch_event_1_reps, cosine_mat_p)
        batch_event_1_reps = batch_event_1_reps.permute((0, 2, 1))
        # vector normalization
        norms_1 = (batch_event_1_reps**2).sum(axis=-1, keepdims=True) ** 0.5
        batch_event_1_reps = batch_event_1_reps / norms_1
        # batch_event_2
        batch_event_2_reps = cosine_ffnn(batch_event_2_reps)
        batch_event_2_reps = batch_event_2_reps.unsqueeze(dim=1)
        batch_event_2_reps = cosine_mat_q * batch_event_2_reps
        batch_event_2_reps = batch_event_2_reps.permute((0, 2, 1))
        batch_event_2_reps = torch.matmul(batch_event_2_reps, cosine_mat_p)
        batch_event_2_reps = batch_event_2_reps.permute((0, 2, 1))
        # vector normalization
        norms_2 = (batch_event_2_reps**2).sum(axis=-1, keepdims=True) ** 0.5
        batch_event_2_reps = batch_event_2_reps / norms_2
        return torch.sum(batch_event_1_reps * batch_event_2_reps, dim=-1)

    # batched_index_select
    def batched_index_select(self, input, dim, index):
        for i in range(1, len(input.shape)):
            if i != dim:
                index = index.unsqueeze(i)
        expanse = list(input.shape)
        expanse[0] = -1
        expanse[dim] = -1
        index = index.expand(expanse)
        return torch.gather(input, dim, index)

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def _matching_func(self, batch_event_1_reps, batch_event_2_reps):
        batch_e1_e2_product = batch_event_1_reps * batch_event_2_reps
        batch_multi_cosine = self.multi_perspective_cosine(
            self.cosine_ffnn, self.cosine_mat_p, self.cosine_mat_q, batch_event_1_reps, batch_event_2_reps
        )
        batch_e1_e2_match = torch.cat([batch_e1_e2_product, batch_multi_cosine], dim=-1)
        return batch_e1_e2_match

    def forward(
        self, batch_inputs, batch_mask_idx, batch_event_idx, label_word_id, batch_mask_inputs=None, labels=None
    ):
        outputs = self.roberta(**batch_inputs)
        self.use_device = self.roberta.device
        sequence_output = outputs.last_hidden_state
        batch_mask_reps = self.batched_index_select(sequence_output, 1, batch_mask_idx.unsqueeze(-1)).squeeze(1)
        if batch_mask_inputs is not None:
            mask_outputs = self.roberta(**batch_mask_inputs)
            mask_sequence_output = mask_outputs.last_hidden_state
            batch_mask_mask_reps = self.batched_index_select(
                mask_sequence_output, 1, batch_mask_idx.unsqueeze(-1)
            ).squeeze(1)
        # extract events & matching
        batch_e1_idx, batch_e2_idx = [], []
        for e1s, e1e, e2s, e2e in batch_event_idx:
            batch_e1_idx.append([[e1s, e1e]])
            batch_e2_idx.append([[e2s, e2e]])
        batch_e1_idx, batch_e2_idx = (
            torch.tensor(batch_e1_idx).to(self.use_device),
            torch.tensor(batch_e2_idx).to(self.use_device),
        )
        batch_event_1_reps = self.span_extractor(sequence_output, batch_e1_idx).squeeze(dim=1)
        batch_event_2_reps = self.span_extractor(sequence_output, batch_e2_idx).squeeze(dim=1)
        batch_match_reps = self._matching_func(batch_event_1_reps, batch_event_2_reps)
        batch_mask_reps = self.mapping(torch.cat([batch_mask_reps, batch_match_reps], dim=-1))
        if batch_mask_inputs is not None:
            batch_mask_event_1_reps = self.span_extractor(mask_sequence_output, batch_e1_idx).squeeze(dim=1)
            batch_mask_event_2_reps = self.span_extractor(mask_sequence_output, batch_e2_idx).squeeze(dim=1)
            batch_mask_match_reps = self._matching_func(batch_mask_event_1_reps, batch_mask_event_2_reps)
            batch_mask_mask_reps = self.mapping(torch.cat([batch_mask_mask_reps, batch_mask_match_reps], dim=-1))
        logits = self.lm_head(batch_mask_reps)[:, label_word_id]
        if batch_mask_inputs is not None:
            mask_logits = self.lm_head(batch_mask_mask_reps)[:, label_word_id]

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            if batch_mask_inputs is not None:
                loss = 0.5 * loss + 0.5 * loss_fct(mask_logits, labels)
        return loss, logits


class RobertaForBasePromptModule(pl.LightningModule):
    def __init__(self, conf, len_token, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(conf)
        self.validation_step_outputs = []
        config = AutoConfig.from_pretrained(conf["transformer_model"])
        self.lm_head = RobertaLMHead(config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.ecr_model = RobertaForBasePrompt(config=config, roberta=self.roberta, lm_head=self.lm_head)
        self.ecr_model.resize_token_embeddings(len_token)

    def forward(
        self,
        batch_inputs,
        batch_mask_idx,
        batch_event_idx,
        label_word_id,
        batch_mask_inputs=None,
        labels=None,
        batch_mention_ids=None,
    ):
        loss, logits = self.ecr_model.forward(
            batch_inputs, batch_mask_idx, batch_event_idx, label_word_id, batch_mask_inputs, labels
        )
        logits = torch.softmax(logits, dim=-1)
        similarities = logits[:, 1]
        distances = 1 - similarities
        distances = distances
        distances_square = torch.square(distances)
        result = {}
        result["distances"] = distances
        result["distances_square"] = distances_square
        result["loss"] = loss
        result["logits"] = logits
        result["labels"] = labels
        return result

    def training_step(self, batch, batch_idx):
        if isinstance(batch, list):
            batch = batch[0]
        forward_output = self.forward(**batch)
        if forward_output["loss"] is not None:
            return forward_output["loss"]
        else:
            return None

    def on_train_epoch_end(self):
        torch.cuda.empty_cache()

    def validation_step(self, batch, batch_idx: int):
        if isinstance(batch, list):
            batch = batch[0]
        forward_output = self.forward(**batch)
        if forward_output["loss"] is not None:
            return forward_output["loss"]
        else:
            return None

    def on_validation_epoch_end(self):
        torch.cuda.empty_cache()

    def get_optimizer_and_scheduler(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {"params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

        if self.hparams.optimizer == "adam":
            optimizer = AdamW(
                optimizer_grouped_parameters,
                self.hparams.learning_rate,
                betas=(self.hparams.adam_beta1, self.hparams.adam_beta2),
                eps=self.hparams.adam_epsilon,
            )
        else:
            raise NotImplementedError

        lr_scheduler = get_scheduler("linear", optimizer, num_warmup_steps=500, num_training_steps=300000)

        return optimizer, lr_scheduler

    def configure_optimizers(self):
        optimizer, lr_scheduler = self.get_optimizer_and_scheduler()
        return [optimizer], [lr_scheduler]

    def predict_step(self, batch, batch_idx: int):
        if isinstance(batch, list):
            batch = batch[0]
        result = self(**batch)
        return result


class CorefPromptModel(PlEcrModel):
    def __init__(
        self,
        mode: str,
        model_dir: str,
        model_filename: str,
        trainer_parameters: Dict[str, Any],
        best_model_filename: str = "best.ckpt",
        conf: Optional[DictConfig] = None,
    ):
        super().__init__(mode, model_dir, model_filename, trainer_parameters, best_model_filename, conf)
        seed_everything(42)

    def build_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.conf["module"]["transformer_model"])

        base_sp_tokens = self.get_special_tokens(token_type="base")
        connect_tokens = self.get_special_tokens(token_type="connect")
        sp_tokens = base_sp_tokens + connect_tokens if "c" in self.conf["dataset"]["prompt_type"] else base_sp_tokens
        self.tokenizer.add_special_tokens({"additional_special_tokens": sp_tokens})

    def instanciate_module(self):
        module = RobertaForBasePromptModule(self.conf["module"], len_token=len(self.tokenizer))
        return module

    def load_module(self, filepath: str):
        result = RobertaForBasePromptModule.load_from_checkpoint(filepath, len_token=len(self.tokenizer))
        return result

    def get_special_tokens(self, token_type: str):
        assert token_type in ["base", "connect", "match"]
        if token_type == "base":
            return [
                "<e1_start>",
                "<e1_end>",
                "<e2_start>",
                "<e2_end>",
                "<l1>",
                "<l2>",
                "<l3>",
                "<l4>",
                "<l5>",
                "<l6>",
                "<l7>",
                "<l8>",
                "<l9>",
                "<l10>",
            ]
        elif token_type == "connect":
            return ["<refer_to>", "<not_refer_to>"]
        elif token_type == "match":
            return ["<match>", "<mismatch>"]
        else:
            raise NotImplementedError

    def create_verbalizer(self, tokenizer: RobertaTokenizer, prompt_type: str):
        base_verbalizer = (
            {
                "coref": {"token": "same", "id": tokenizer.convert_tokens_to_ids("same")},
                "non-coref": {"token": "different", "id": tokenizer.convert_tokens_to_ids("different")},
            }
            if prompt_type.startswith("ma")
            else {
                "coref": {
                    "token": "<refer_to>",
                    "id": tokenizer.convert_tokens_to_ids("<refer_to>"),
                    "description": "refer to",
                }
                if "c" in prompt_type
                else {"token": "yes", "id": tokenizer.convert_tokens_to_ids("yes")}
                if "q" in prompt_type
                else {"token": "same", "id": tokenizer.convert_tokens_to_ids("same")},
                "non-coref": {
                    "token": "<not_refer_to>",
                    "id": tokenizer.convert_tokens_to_ids("<not_refer_to>"),
                    "description": "not refer to",
                }
                if "c" in prompt_type
                else {"token": "no", "id": tokenizer.convert_tokens_to_ids("no")}
                if "q" in prompt_type
                else {"token": "different", "id": tokenizer.convert_tokens_to_ids("different")},
            }
        )

        if prompt_type.startswith("h") or prompt_type.startswith("s"):  # base prompt
            return base_verbalizer
        else:  # mix prompt
            if prompt_type != "ma_remove-anchor":
                for subtype, s_id in subtype2id.items():
                    base_verbalizer[subtype] = {
                        "token": f"<st_{s_id}>",
                        "id": tokenizer.convert_tokens_to_ids(f"<st_{s_id}>"),
                        "description": subtype if subtype != "other" else "normal",
                    }
            if prompt_type != "ma_remove-match":
                base_verbalizer["match"] = {
                    "token": "<match>",
                    "id": tokenizer.convert_tokens_to_ids("<match>"),
                    "description": "same related relevant similar matching matched",
                }
                base_verbalizer["mismatch"] = {
                    "token": "<mismatch>",
                    "id": tokenizer.convert_tokens_to_ids("<mismatch>"),
                    "description": "different unrelated irrelevant dissimilar mismatched",
                }
            return base_verbalizer

    def collate_fn(self, batch_samples):
        verbalizer = self.create_verbalizer(self.tokenizer, self.conf["dataset"]["prompt_type"])
        pos_id, neg_id = verbalizer["coref"]["id"], verbalizer["non-coref"]["id"]
        tokenizer = self.tokenizer
        batch_sen, batch_mask_idx, batch_event_idx, batch_labels, batch_mention_ids = [], [], [], [], []
        for sample in batch_samples:
            batch_sen.append(sample["prompt"])
            # convert char offsets to token idxs
            encoding = tokenizer(sample["prompt"])
            mask_idx = encoding.char_to_token(sample["mask_offset"])
            e1s_idx, e1e_idx, e2s_idx, e2e_idx = (
                encoding.char_to_token(sample["e1s_offset"]),
                encoding.char_to_token(sample["e1e_offset"]),
                encoding.char_to_token(sample["e2s_offset"]),
                encoding.char_to_token(sample["e2e_offset"]),
            )
            assert None not in [mask_idx, e1s_idx, e1e_idx, e2s_idx, e2e_idx]
            batch_mask_idx.append(mask_idx)
            batch_event_idx.append([e1s_idx, e1e_idx, e2s_idx, e2e_idx])
            batch_labels.append(int(sample["label"]))
            batch_mention_ids.append((sample["mention1_id"], sample["mention2_id"]))
        batch_inputs = tokenizer(
            batch_sen,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return {
            "batch_inputs": {k_: v_ for k_, v_ in batch_inputs.items()},
            "batch_mask_idx": torch.tensor(batch_mask_idx),
            "batch_event_idx": batch_event_idx,
            "label_word_id": [neg_id, pos_id],
            "labels": torch.tensor(batch_labels),
            "batch_mention_ids": batch_mention_ids,
        }

    def prepare_data(self, data: EcrData, mode: str):
        if data is None:
            return None, None
        if mode == "train":
            if len(data.documents) > 100:
                dataset = CorefPromptCoreferenceDataset(
                    ecr_data=data,
                    tokenizer=self.tokenizer,
                    simi_file=self.conf["dataset"]["train_simi_file"],
                    prompt_type=self.conf["dataset"]["prompt_type"],
                    select_arg_strategy=self.conf["dataset"]["select_arg_strategy"],
                    max_length=512,
                    mode=mode,
                )
            else:
                dataset = CorefPromptCoreferenceDataset(
                    ecr_data=data,
                    tokenizer=self.tokenizer,
                    simi_file=self.conf["dataset"]["dev_simi_file"],
                    prompt_type=self.conf["dataset"]["prompt_type"],
                    select_arg_strategy=self.conf["dataset"]["select_arg_strategy"],
                    max_length=512,
                    mode=mode,
                )
        else:
            if len(data.documents) < 100:
                dataset = CorefPromptCoreferenceDataset(
                    ecr_data=data,
                    tokenizer=self.tokenizer,
                    simi_file=self.conf["dataset"]["dev_simi_file"],
                    prompt_type=self.conf["dataset"]["prompt_type"],
                    select_arg_strategy=self.conf["dataset"]["select_arg_strategy"],
                    max_length=512,
                    mode=mode,
                )
            else:
                dataset = CorefPromptCoreferenceDataset(
                    ecr_data=data,
                    tokenizer=self.tokenizer,
                    simi_file=self.conf["dataset"]["test_simi_file"],
                    prompt_type=self.conf["dataset"]["prompt_type"],
                    select_arg_strategy=self.conf["dataset"]["select_arg_strategy"],
                    max_length=512,
                    mode=mode,
                )

        batch_size = self.conf["dataloader"][f"{mode}_batch_size"]
        num_workers = self.conf["dataloader"]["num_workers"]
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=self.collate_fn,
            shuffle=(mode == "train"),
            num_workers=num_workers,
            pin_memory=True,
        )
        dataloaders = [dataloader]
        return dataset, dataloaders

    def inner_pred(self, trainer, module, dataloaders, dataset):
        predictions = trainer.predict(module, dataloaders=dataloaders)
        distances = []
        for prediction in predictions:
            distances.extend(prediction["distances"].numpy().tolist())

        data = dataset.data
        id_id_distance = {}

        for i, d in enumerate(data):
            mention1_id, mention2_id = d["mention1_id"], d["mention2_id"]
            if mention1_id not in id_id_distance:
                id_id_distance[mention1_id] = {}
            if mention2_id not in id_id_distance:
                id_id_distance[mention2_id] = {}
            id_id_distance[mention1_id][mention2_id] = distances[i]
            id_id_distance[mention2_id][mention1_id] = distances[i]

        return id_id_distance

    def predict(self, data: EcrData, output_tag: str) -> EcrData:
        trainer = pl.Trainer(
            accelerator="gpu",
            # devices=args.devices
        )

        test_dataset, test_dataloaders = self.prepare_data(data, mode="predict")
        id_id_distance = self.inner_pred(trainer, self.module, test_dataloaders, test_dataset)
        documents = {}
        doc_mentions = {}
        events = []
        for mention_id, id_distance in id_id_distance.items():
            mention = data.mentions[mention_id]
            mention.add_tag(output_tag, id_distance)
            doc_mentions[mention_id] = mention
            doc_id = mention.doc_id
            document = data.documents[doc_id]
            documents[doc_id] = document

        event2mentions = defaultdict(list)
        for _, mention in doc_mentions.items():
            event_id = mention.meta["event_id"]
            event2mentions[event_id].append(mention)
        for event_id, mentions in event2mentions.items():
            event = Event(event_id, mentions)
            events.append(event)

        data = EcrData(data.name, documents, doc_mentions, events, data.meta)
        return data

    def get_predict_type(self) -> str:
        result = Mention.mention_distance_tag_name
        return result
