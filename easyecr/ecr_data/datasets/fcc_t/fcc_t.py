#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""



Date: 2023/11/20 10:00
"""
import os
from collections import defaultdict

import pandas as pd
from pandas import DataFrame

from easyecr.ecr_data.data_structure.data_structure import Document
from easyecr.ecr_data.data_structure.data_structure import Mention
from easyecr.ecr_data.data_structure.data_structure import Event
from easyecr.ecr_data.data_structure.data_structure import EcrData
from easyecr.ecr_data.data_structure.data_structure import RichText
from easyecr.ecr_data.datasets.base_ecr_dataset.base_ecr_dataset import BaseEcrDataset


"""
with stacked
Total number of documents is (207 + 117 + 127 = 451), matching with golden 451;
Total number of mentions is (1604 + 750 + 1209 = 3563), matching with golden 3563;
Total number of events is (259 + 136 + 191 = 586), larger than golden 469;

without stacked
Total number of documents is (207+117+127 = 451), matching with golden 451;
Total number of mentions is (1469+680+1074 = 3223), smaller than golden 3563;
Total number of events is (259+136+191 = 514), larger than golden 469;

Dataset golden statistics from Paper: Generalizing Cross-Document Event Coreference Resolution Across Multiple Corpora
"""
FCC_TOTAL_NUM_OF_GOLDEN_DOCUMENTS = 451
FCC_TOTAL_NUM_OF_GOLDEN_MENTIONS = 3563
FCC_TOTAL_NUM_OF_GOLDEN_EVENTS = 469
# with stacked
FCC_TRAIN_WITHSTACK_NUM_OF_DOCUMENTS = 207
FCC_TRAIN_WITHSTACK_NUM_OF_MENTIONS = 1604
FCC_TRAIN_WITHSTACK_NUM_OF_EVENTS = 259
FCC_DEV_WITHSTACK_NUM_OF_DOCUMENTS = 117
FCC_DEV_WITHSTACK_NUM_OF_MENTIONS = 750
FCC_DEV_WITHSTACK_NUM_OF_EVENTS = 136
FCC_TEST_WITHSTACK_NUM_OF_DOCUMENTS = 127
FCC_TEST_WITHSTACK_NUM_OF_MENTIONS = 1209
FCC_TEST_WITHSTACK_NUM_OF_EVENTS = 191
# without stacked
FCC_TRAIN_WITHOUTSTACK_NUM_OF_DOCUMENTS = 207
FCC_TRAIN_WITHOUTSTACK_NUM_OF_MENTIONS = 1469
FCC_TRAIN_WITHOUTSTACK_NUM_OF_EVENTS = 236
FCC_DEV_WITHOUTSTACK_NUM_OF_DOCUMENTS = 117
FCC_DEV_WITHOUTSTACK_NUM_OF_MENTIONS = 680
FCC_DEV_WITHOUTSTACK_NUM_OF_EVENTS = 111
FCC_TEST_WITHOUTSTACK_NUM_OF_DOCUMENTS = 127
FCC_TEST_WITHOUTSTACK_NUM_OF_MENTIONS = 1074
FCC_TEST_WITHOUTSTACK_NUM_OF_EVENTS = 167


class FCCTDataset(BaseEcrDataset):
    def __init__(
        self,
        dataset_name: str,
        directory: str,
        stacked: bool = False,
    ):
        super().__init__(dataset_name, directory)
        if stacked:
            self.file_dir = os.path.join(directory, f"with_stacked_actions")
        else:
            self.file_dir = os.path.join(directory, f"without_stacked_actions")

        self.documents = {}
        self.load_documents()
        self.mentions = {}
        self.load_mentions()
        self.events = []
        self.load_events()

    def _open_file(self, file_path):
        return pd.read_csv(file_path)

    def parse_text_from_df(self, doc_content_df: DataFrame) -> str:
        doc_info = doc_content_df[["sentence-idx", "token-idx", "token"]]
        doc_info = doc_info.reset_index(drop=True)
        text = []
        doc_token = [
            {"t_id": global_token_idx, "sentence": row[0], "number": row[1], "text": str(row[2])}
            for global_token_idx, row in doc_info.iterrows()
        ]
        for item in doc_token:
            text.append(item["text"])
        text = " ".join(text)
        return text, doc_token

    def load_documents(self):
        docs_csv_path = os.path.join(self.directory, "documents.csv")
        tokens_csv_path = os.path.join(self.directory, "tokens.csv")
        docs_meta_df = self._open_file(docs_csv_path)
        docs_token_df = self._open_file(tokens_csv_path)
        for _, doc_meta_info in docs_meta_df.iterrows():
            doc_id = doc_meta_info["doc-id"]
            doc_pub_date = doc_meta_info["publish-date"]
            doc_collection = doc_meta_info["collection"]
            doc_seminal_event = doc_meta_info["seminal-event"]
            doc_content_df = docs_token_df[docs_token_df["doc-id"] == doc_id]
            text, doc_token = self.parse_text_from_df(doc_content_df)
            meta = {
                "doc_pub_date": doc_pub_date,
                "doc_collection": doc_collection,
                "doc_topic": "1",
                "doc_subtopic": doc_seminal_event,
                "doc_token": doc_token,
            }
            document = Document(doc_id=doc_id, text=text, meta=meta)
            self.documents[doc_id] = document

    def parse_extent(self, doc_content_df: DataFrame, sent_idx: int) -> RichText:
        start = 0
        for i in range(sent_idx):
            start += len(doc_content_df[doc_content_df["sentence-idx"] == i])
        doc_mention_df = doc_content_df[doc_content_df["sentence-idx"] == sent_idx]
        words = list(map(str, doc_mention_df["token"]))
        text = " ".join(map(str, words))
        end = start + doc_mention_df.iloc[-1]["token-idx"]
        return RichText(text=text, start=start, end=end, words=words)

    def parse_anchor(self, extent_words: list, mention_info: DataFrame) -> RichText:
        start = mention_info["token-idx-from"]
        end = mention_info["token-idx-to"]
        words = extent_words[start:end]
        text = " ".join(words)
        return RichText(text=text, start=start, end=end - 1, words=words)

    def parse_argument_for_mentions(self, doc_tokens_df: DataFrame):
        role_path = os.path.join(self.file_dir, f"cross_subtopic_semantic_roles.csv")
        mention2arg = defaultdict(dict)
        for _, role_info in self._open_file(role_path).iterrows():
            doc_id = role_info["doc-id"]
            mention_id = doc_id + "_" + str(role_info["mention-id"])
            role_type = role_info["mention-type-coarse"]
            role_mention_id = role_info["component-mention-id"]
            type_file_path = os.path.join(self.file_dir, f"cross_subtopic_mentions_{role_type}.csv")
            role_df = self._open_file(type_file_path)
            mention_role_df = role_df[(role_df["doc-id"] == doc_id) & (role_df["mention-id"] == role_mention_id)]
            mention_role_sent_idx = mention_role_df["sentence-idx"].iloc[0]

            doc_content_df = doc_tokens_df[doc_tokens_df["doc-id"] == doc_id]
            #
            start = mention_role_df["token-idx-from"].iloc[0]
            end = mention_role_df["token-idx-to"].iloc[0]
            mention_tokens_df = doc_content_df[doc_content_df["sentence-idx"] == mention_role_sent_idx]["token"]
            text = " ".join([mention_tokens_df.iloc[i] for i in range(start, end)])
            words = text.split()
            mention2arg[mention_id].update({role_type: RichText(text=text, start=start, end=end, words=words)})

        for m_id, mention in self.mentions.items():
            mention.arguments = mention2arg[m_id]

    def load_mentions(self):
        action_csv_path = os.path.join(self.file_dir, "cross_subtopic_mentions_action.csv")
        tokens_csv_path = os.path.join(self.directory, "tokens.csv")
        docs_token_df = self._open_file(tokens_csv_path)
        for _, mention_info in self._open_file(action_csv_path).iterrows():
            doc_id = mention_info["doc-id"]
            original_mention_id = mention_info["mention-id"]
            mention_id = doc_id + "_" + str(original_mention_id)
            sent_idx = mention_info["sentence-idx"]
            event_id = mention_info["event"]
            doc_content_df = docs_token_df[docs_token_df["doc-id"] == doc_id]
            extent = self.parse_extent(doc_content_df=doc_content_df, sent_idx=sent_idx)
            anchor = self.parse_anchor(extent_words=extent.words, mention_info=mention_info)
            anchor.start = extent.start + anchor.start
            anchor.end = extent.start + anchor.end
            arguments = None
            meta = {"event_id": event_id, "original_mention_id": original_mention_id}
            mention = Mention(
                doc_id=doc_id, mention_id=mention_id, extent=extent, anchor=anchor, arguments=arguments, meta=meta
            )
            self.mentions[mention_id] = mention
        # set arguments for mention
        self.parse_argument_for_mentions(docs_token_df)

    def load_events(self):
        """Merge the mentions into an event based on the mention_id"""
        event2mentions = defaultdict(list)
        for _, mention in self.mentions.items():
            event_id = mention.meta["event_id"]
            event2mentions[event_id].append(mention)
        for event_id, mentions in event2mentions.items():
            event = Event(event_id, mentions)
            self.events.append(event)

    def to_ecr_data(self, verbose: bool = False) -> EcrData:
        if verbose:
            print(
                f"Statistics of the {self.dataset_name} {self.selection} dataset:\n"
                f"number of documents: {len(self.documents)}\t\n"
                f"number of mentions: {len(self.mentions)}\t\n"
                f"number of events: {len(self.events)}\t"
            )
        return EcrData(
            name=self.dataset_name,
            documents=self.documents,
            mentions=self.mentions,
            events=self.events,
            meta={"index_type": "word"},
        )
