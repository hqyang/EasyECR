import os
import json
from typing import List
from typing import Dict

from easyecr.ecr_data.data_structure.data_structure import Document
from easyecr.ecr_data.data_structure.data_structure import Mention
from easyecr.ecr_data.data_structure.data_structure import Event
from easyecr.ecr_data.data_structure.data_structure import EcrData
from easyecr.ecr_data.data_structure.data_structure import RichText
from easyecr.ecr_data.datasets.base_ecr_dataset.base_ecr_dataset import BaseEcrDataset

"""
Total number of Eng documents is (167), matching with golden 167;
Total number of Eng mentions is (4375), matching with golden 4375;
Total number of Eng events is (2963), larger than golden 2963;

Dataset golden statistics from Github: https://github.com/jsksxs360/event-coref-emnlp2022/tree/main
"""

KBPENG2017_NUM_OF_DOCUMENTS = 167
KBPENG2017_NUM_OF_MENTIONS = 4375
KBPENG2017_NUM_OF_EVENTS = 2963


class KBPEng2017Dataset(BaseEcrDataset):
    def __init__(self, dataset_name: str, directory: str):
        super().__init__(dataset_name, directory)
        self.file_path = os.path.join(directory, f"kbp_eng.json")
        self.documents = {}
        self.load_documents()
        self.mentions = {}
        self.events = []
        self.load_mentions_and_events()

    def _open_file(self):
        with open(self.file_path, "r", encoding="utf-8") as file:
            return file.readlines()

    def load_documents(self):
        raw_data = self._open_file()
        for line in raw_data:
            doc_info = json.loads(line.strip())
            doc_id = doc_info["doc_id"]
            text = doc_info["document"]
            meta = {}
            document = Document(doc_id=doc_id, text=text, meta=meta)
            self.documents[doc_id] = document

    def parse_extent_from_mention(self, mention_info: dict, doc_sentences: List[Dict]) -> RichText:
        sent_idx = mention_info["sent_idx"]
        mention_text = doc_sentences[sent_idx]["text"]
        mention_start = doc_sentences[sent_idx]["start"]
        mention_end = mention_start + len(mention_text) - 1
        return RichText(text=mention_text, start=mention_start, end=mention_end)

    def parse_anchor_from_mention(self, mention_info: dict) -> RichText:
        anchor_text = mention_info["trigger"]
        anchor_start = mention_info["sent_start"]
        anchor_end = anchor_start + len(anchor_text) - 1
        return RichText(text=anchor_text, start=anchor_start, end=anchor_end)

    def parse_mentions_from_doc(self, doc_id: str, doc_sentences: List[Dict], doc_mentions: List[Dict]):
        for mention_info in doc_mentions:
            mention_id = (
                doc_id + "_" + mention_info["event_id"]
            )  # The original mention_id is not unique in the entire corpus, so the doc_id is inserted

            extent = self.parse_extent_from_mention(mention_info, doc_sentences)
            anchor = self.parse_anchor_from_mention(mention_info)
            anchor.start = anchor.start + extent.start
            anchor.end = anchor.end + extent.start
            arguments = None
            meta = {
                "mention_type": mention_info["type"],
                "mention_subtype": mention_info["subtype"],
                "mention_realis": mention_info["realis"],
                "original_mention_id": mention_info["event_id"],
            }
            mention = Mention(
                doc_id=doc_id, mention_id=mention_id, extent=extent, anchor=anchor, arguments=arguments, meta=meta
            )
            self.mentions[mention_id] = mention

    def parse_events_from_doc(self, doc_id: str, doc_events: List[Dict]):
        for event_info in doc_events:
            event_id = event_info["hopper_id"]
            mention_ids = event_info["events"]
            mentions = []
            for m_id in mention_ids:
                mention = self.mentions[doc_id + "_" + m_id]
                mention_id = mention.mention_id
                mention.meta.update({"event_id": event_id})
                self.mentions[mention_id] = mention
                mentions.append(mention)
            event = Event(event_id=event_id, mentions=mentions)
            self.events.append(event)

    def load_mentions_and_events(self):
        raw_data = self._open_file()
        for line in raw_data:
            doc_info = json.loads(line.strip())
            doc_id = doc_info["doc_id"]
            doc_sentences = doc_info["sentences"]
            doc_mentions = doc_info["events"]
            self.parse_mentions_from_doc(doc_id=doc_id, doc_sentences=doc_sentences, doc_mentions=doc_mentions)
            doc_events = doc_info["clusters"]
            self.parse_events_from_doc(doc_id=doc_id, doc_events=doc_events)

    def to_ecr_data(self, verbose: bool = False) -> EcrData:
        if verbose:
            print(
                f"Statistics of the {self.dataset_name} dataset:\n"
                f"number of documents: {len(self.documents)}\t number of golden documents: {KBPENG2017_NUM_OF_DOCUMENTS}\n"
                f"number of mentions: {len(self.mentions)}\t golden number of mentions: {KBPENG2017_NUM_OF_MENTIONS}\n"
                f"number of events: {len(self.events)}\t number of golden events: {KBPENG2017_NUM_OF_EVENTS}"
            )
        return EcrData(
            name=self.dataset_name,
            documents=self.documents,
            mentions=self.mentions,
            events=self.events,
            meta={"index_type": "char"},
        )
