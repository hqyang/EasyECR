#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""



Date: 2023/11/20 10:13
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
Total number of documents is (207 + 117 + 127 = 451), matching with the golden 451;
Total number of mentions is (1195 + 535 +888 = 2618), matching with the golden 2374;
Total number of events is (115 + 47 + 55 = 217), matching with the golden 218;

Dataset golden statistics from Paper: Breaking the Subtopic Barrier in Cross-Document Event Coreference Resolution
"""
FCC_GOLDEN_NUM_OF_DOCUMENTS = 451
FCC_GOLDEN_NUM_OF_MENTIONS = 2374
FCC_GOLDEN_NUM_OF_EVENTS = 218
FCC_TRAIN_NUM_OF_DOCUMENTS = 207
FCC_TRAIN_NUM_OF_MENTIONS = 1195
FCC_TRAIN_NUM_OF_EVENTS = 115
FCC_DEV_NUM_OF_DOCUMENTS = 117
FCC_DEV_NUM_OF_MENTIONS = 535
FCC_DEV_NUM_OF_EVENTS = 47
FCC_TEST_NUM_OF_DOCUMENTS = 127
FCC_TEST_NUM_OF_MENTIONS = 888
FCC_TEST_NUM_OF_EVENTS = 55


class FCCDataset(BaseEcrDataset):
    def __init__(self, dataset_name: str, directory: str):
        super().__init__(dataset_name, directory)

        self.documents = {}
        self.load_documents()
        self.mentions = {}
        self.load_mentions()
        self.events = []
        self.load_events()

    def _open_file(self, file_path):
        raw_data = pd.read_csv(file_path)
        return raw_data

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

    def parse_extent_or_anchor(self, doc_content_df: DataFrame, sent_idx: int) -> RichText:
        start = 0
        for i in range(sent_idx):
            start += len(doc_content_df[doc_content_df["sentence-idx"] == i])
        doc_mention_df = doc_content_df[doc_content_df["sentence-idx"] == sent_idx]
        words = list(doc_mention_df["token"])
        text = " ".join(map(str, words))
        end = start + doc_mention_df.iloc[-1]["token-idx"]
        result = RichText(text=text, start=start, end=end, words=words)
        return result

    def load_mentions(self):
        mentions_csv_path = os.path.join(self.directory, "mentions_cross_subtopic.csv")
        docs_mentions_df = self._open_file(mentions_csv_path)
        tokens_csv_path = os.path.join(self.directory, "tokens.csv")
        docs_token_df = self._open_file(tokens_csv_path)
        for _, mention_info in docs_mentions_df.iterrows():
            doc_id = mention_info["doc-id"]
            original_mention_id = mention_info["mention-id"]
            mention_id = doc_id + "_" + str(original_mention_id)
            event_id = mention_info["event"]
            sent_idx = mention_info["sentence-idx"]
            doc_content_df = docs_token_df[docs_token_df["doc-id"] == doc_id]
            extent = self.parse_extent_or_anchor(doc_content_df=doc_content_df, sent_idx=sent_idx)
            anchor = self.parse_extent_or_anchor(doc_content_df=doc_content_df, sent_idx=sent_idx)
            meta = {"event_id": event_id, "original_mention_id": original_mention_id}
            mention = Mention(
                doc_id=doc_id, mention_id=mention_id, extent=extent, anchor=anchor, arguments=None, meta=meta
            )
            self.mentions[mention_id] = mention

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
                f"Statistics of the {self.dataset_name} dataset:\n"
                f"number of documents: {len(self.documents)}\t\n"
                f"number of mentions: {len(self.mentions)}\t\n"
                f"number of events: {len(self.events)}\t"
            )
        return EcrData(
            name=self.dataset_name,
            documents=self.documents,
            mentions=self.mentions,
            events=self.events,
            meta={"index_type": "word", "trigger_level": "sentence"},
        )
