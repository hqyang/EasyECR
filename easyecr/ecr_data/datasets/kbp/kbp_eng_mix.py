import json
from typing import Dict
from typing import List

import spacy
from spacy.language import Language

from easyecr.ecr_data.data_structure.data_structure import Document
from easyecr.ecr_data.data_structure.data_structure import Mention
from easyecr.ecr_data.data_structure.data_structure import Event
from easyecr.ecr_data.data_structure.data_structure import EcrData
from easyecr.ecr_data.data_structure.data_structure import RichText
from easyecr.ecr_data.datasets.base_ecr_dataset.base_ecr_dataset import BaseEcrDataset

"""
Except that the number of mentions in the training set is three less than that of golden, other statistics are matching.
Note: KBPEngMixDataset, which Test consists of KBP2017, Train and Dev are the results of re-division after mixing KBP2015 and 2016.

Statistics of the kbpengmix train dataset:
number of documents: 735         number of golden documents: 735
number of mentions: 20509        golden number of mentions: 20512
number of events: 13292  number of golden events: 13292
Statistics of the kbpengmix dev dataset:
number of documents: 82  number of golden documents: 82
number of mentions: 2382         golden number of mentions: 2382
number of events: 1502   number of golden events: 1502
Statistics of the kbpengmix test dataset:
number of documents: 167         number of golden documents: 167
number of mentions: 4375         golden number of mentions: 4375
number of events: 2963   number of golden events: 2963

Dataset golden statistics from Paper: Conundrums in Event Coreference Resolution: Making Sense of the State of the Art
"""
KBPENGMIX_TRAIN_NUM_OF_DOCUMENTS = 735
KBPENGMIX_TRAIN_NUM_OF_MENTIONS = 20512
KBPENGMIX_TRAIN_NUM_OF_EVENTS = 13292
KBPENGMIX_DEV_NUM_OF_DOCUMENTS = 82
KBPENGMIX_DEV_NUM_OF_MENTIONS = 2382
KBPENGMIX_DEV_NUM_OF_EVENTS = 1502
KBPENGMIX_TEST_NUM_OF_DOCUMENTS = 167
KBPENGMIX_TEST_NUM_OF_MENTIONS = 4375
KBPENGMIX_TEST_NUM_OF_EVENTS = 2963


class KBPEngMixDataset(BaseEcrDataset):
    def __init__(self, dataset_name: str, directory: str):
        super().__init__(dataset_name, directory)
        self.documents = {}
        self.load_documents()
        self.mentions = {}
        self.events = []
        self.load_mentions_and_events()

    def _open_file(self):
        with open(self.directory, "r", encoding="utf-8") as file:
            return file.readlines()

    def parse_doc_token(self, nlp: Language, doc_sentences: List):
        doc_token = []
        global_token_idx = 0
        for i, sent in enumerate(doc_sentences):
            sent_i = i
            doc = nlp(sent["text"])
            for j, token in enumerate(doc):
                text = token.text
                doc_token.append({"t_id": global_token_idx, "sentence": sent_i, "number": j, "text": text})
                global_token_idx += 1
        return doc_token

    def load_documents(self):
        nlp = spacy.load("en_core_web_sm")
        raw_data = self._open_file()
        for i, line in enumerate(raw_data):
            doc_info = json.loads(line.strip())
            doc_id = doc_info["doc_id"]
            text = doc_info["document"]
            doc_token = self.parse_doc_token(nlp, doc_info["sentences"])
            meta = {"sentences": doc_info["sentences"], "doc_token": doc_token}
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
                "type": mention_info["type"],
                "subtype": mention_info["subtype"],
                "realis": mention_info["realis"],
                "original_mention_id": mention_info["event_id"],
                "sent_idx": mention_info["sent_idx"],
            }
            mention = Mention(
                doc_id=doc_id, mention_id=mention_id, extent=extent, anchor=anchor, arguments=arguments, meta=meta
            )
            self.mentions[mention_id] = mention

    def parse_events_from_doc(self, doc_id: str, doc_events: List[Dict]):
        for event_info in doc_events:
            event_id = event_info["hopper_id"]
            mention_ids = event_info["events"]
            # mentions = [self.mentions[doc_id + "_" + m_id] for m_id in mention_ids]
            mentions = []
            for m_id in mention_ids:
                mention = self.mentions[doc_id + "_" + m_id]
                mention.meta.update({"event_id": event_id})
                mentions.append(mention)
            event = Event(event_id=event_id, mentions=mentions)
            self.events.append(event)

    def load_mentions_and_events(self):
        raw_data = self._open_file()
        for i, line in enumerate(raw_data):
            doc_info = json.loads(line.strip())
            doc_id = doc_info["doc_id"]
            doc_sentences = doc_info["sentences"]
            doc_mentions = doc_info["events"]
            self.parse_mentions_from_doc(doc_id=doc_id, doc_sentences=doc_sentences, doc_mentions=doc_mentions)
            doc_events = doc_info["clusters"]
            self.parse_events_from_doc(doc_id=doc_id, doc_events=doc_events)

    def to_ecr_data(self, verbose=False) -> EcrData:
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
            meta={"index_type": "char"},
        )
