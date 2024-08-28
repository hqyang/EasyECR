import json
from typing import List
from typing import Dict
from collections import defaultdict


import spacy
from spacy.language import Language

from easyecr.ecr_data.data_structure.data_structure import Document
from easyecr.ecr_data.data_structure.data_structure import Mention
from easyecr.ecr_data.data_structure.data_structure import Event
from easyecr.ecr_data.data_structure.data_structure import EcrData
from easyecr.ecr_data.data_structure.data_structure import RichText
from easyecr.ecr_data.datasets.base_ecr_dataset.base_ecr_dataset import BaseEcrDataset


"""
Total number of Eng documents is (2913+710+857=4480), matching with golden 4480;
Total number of Eng mentions is (73939+17780+20557=112276), matching with golden 112276;
Total number of Eng events is (67984+16301+18908=103193), matching with golden 103193;

Dataset golden statistics from Paper: MAVEN-ERE: A Unified Large-scale Dataset for Event Coreference, Temporal, Causal, and Subevent Relation Extraction
"""

MAVENERE_TRAIN_NUM_OF_DOCUMENTS = 2913
MAVENERE_DEV_NUM_OF_DOCUMENTS = 710
MAVENERE_TEST_NUM_OF_DOCUMENTS = 857
MAVENERE_TRAIN_NUM_OF_MENTIONS = 73939
MAVENERE_DEV_NUM_OF_MENTIONS = 17780
MAVENERE_TEST_NUM_OF_MENTIONS = 20557
MAVENERE_TRAIN_NUM_OF_EVENTS = 67984
MAVENERE_DEV_NUM_OF_EVENTS = 16301
MAVENERE_TEST_NUM_OF_EVENTS = 18908


class MAVENEREDataset(BaseEcrDataset):
    def __init__(self, dataset_name: str, directory: str):
        super().__init__(dataset_name, directory)
        self.directory = directory
        self.documents = {}
        self.load_documents()
        self.mentions = {}
        self.load_mentions()
        self.events = None
        if "train" in directory or "valid" in directory:
            self.load_events()

    def _open_file(self) -> List[str]:
        with open(self.directory, "r") as file:
            return file.readlines()

    def parse_doc_token(self, title_tokens: List, doc_tokens: List):
        doc_token = []
        global_token_idx = 0
        for i, token in enumerate(title_tokens):
            doc_token.append({"t_id": global_token_idx, "sentence": 0, "number": i, "text": token})
            global_token_idx += 1
        for j, item in enumerate(doc_tokens, start=1):
            for k, token in enumerate(item):
                doc_token.append({"t_id": global_token_idx, "sentence": j, "number": k, "text": token})
                global_token_idx += 1
        return doc_token

    def load_documents(self):
        nlp = spacy.load("en_core_web_sm")

        raw_data = self._open_file()
        for i, line in enumerate(raw_data):
            doc_info = json.loads(line.strip())
            doc_id = doc_info["id"]
            doc_title = doc_info["title"]
            doc_sentences = doc_info["sentences"]

            doc_tokens = doc_info["tokens"]
            text = doc_title
            for sentence in doc_sentences:
                text += f" {sentence}"
            title_tokens = [token.text for token in nlp(doc_title)]
            doc_token = self.parse_doc_token(title_tokens, doc_tokens)
            meta = {
                "doc_title": doc_title,
                "doc_token": doc_token,
                "sentences": doc_sentences,
            }
            document = Document(doc_id=doc_id, text=text, meta=meta)
            self.documents[doc_id] = document

    def parse_extent_for_mention(
        self, doc_tokens: List[List[str]], doc_sentences: List[str], title_tokens: List[str], sen_idx: int
    ):
        extent_text = doc_sentences[sen_idx]
        extent_words = doc_tokens[sen_idx]
        extent_start = len(title_tokens)
        for i in range(sen_idx):
            extent_start += len(doc_tokens[i])
        extent_end = extent_start + len(extent_words)
        return RichText(text=extent_text, start=extent_start, end=extent_end - 1, words=extent_words)

    def parse_anchor_for_mention(self, trigger_word: str, trigger_offset: list):
        anchor_text = trigger_word
        anchor_start, anchor_end = trigger_offset
        anchor_words = trigger_word.split(" ")
        assert anchor_end - anchor_start == len(anchor_words)
        return RichText(text=anchor_text, start=anchor_start, end=anchor_end - 1, words=anchor_words)

    def parse_mentions_from_doc(self, doc_info: Dict, lang: Language):
        doc_id = doc_info["id"]
        doc_title = doc_info["title"]
        title_tokens = [token.text for token in lang(doc_title)]
        doc_tokens = doc_info["tokens"]
        doc_sentences = doc_info["sentences"]

        if "test" in self.directory:  # test.jsonl without ground truth
            doc_event_mentions = doc_info["event_mentions"]
            for mention_info in doc_event_mentions:
                mention_id = mention_info["id"]
                extent = self.parse_extent_for_mention(
                    doc_tokens=doc_tokens,
                    doc_sentences=doc_sentences,
                    title_tokens=title_tokens,
                    sen_idx=mention_info["sent_id"],
                )
                anchor = self.parse_anchor_for_mention(
                    trigger_word=mention_info["trigger_word"], trigger_offset=mention_info["offset"]
                )
                anchor.start = anchor.start + extent.start
                anchor.end = anchor.end + extent.start
                meta = {
                    "event_type": mention_info["type"],
                    "event_type_id": mention_info["type_id"],
                    "sent_idx": mention_info["sent_id"],
                }
                mention = Mention(
                    doc_id=doc_id, mention_id=mention_id, extent=extent, anchor=anchor, arguments=None, meta=meta
                )
                self.mentions[mention_id] = mention
        else:  # train and valid with groud truth
            doc_event_mentions = doc_info["events"]
            for event_info in doc_event_mentions:
                event_id = event_info["id"]
                event_type = event_info["type"]
                event_type_id = event_info["type_id"]
                event_mentions = event_info["mention"]

                for mention_info in event_mentions:
                    mention_id = mention_info["id"]
                    extent = self.parse_extent_for_mention(
                        doc_tokens=doc_tokens,
                        doc_sentences=doc_sentences,
                        title_tokens=title_tokens,
                        sen_idx=mention_info["sent_id"],
                    )
                    anchor = self.parse_anchor_for_mention(
                        trigger_word=mention_info["trigger_word"], trigger_offset=mention_info["offset"]
                    )
                    anchor.start = anchor.start + extent.start
                    anchor.end = anchor.end + extent.start
                    meta = {
                        "event_id": event_id,
                        "event_type": event_type,
                        "event_type_id": event_type_id,
                        "sent_idx": mention_info["sent_id"],
                    }
                    mention = Mention(
                        doc_id=doc_id,
                        mention_id=mention_id,
                        extent=extent,
                        anchor=anchor,
                        arguments=None,
                        meta=meta,
                    )
                    self.mentions[mention_id] = mention

    def load_mentions(self):
        nlp = spacy.load("en_core_web_sm")
        raw_data = self._open_file()
        for i, line in enumerate(raw_data):
            doc_info = json.loads(line.strip())
            self.parse_mentions_from_doc(doc_info=doc_info, lang=nlp)

    def load_events(self):
        """Merge the mentions into an event based on the mention_id"""
        self.events = []
        event2mentions = defaultdict(list)
        for _, mention in self.mentions.items():
            event_id = mention.meta["event_id"]
            event2mentions[event_id].append(mention)
        for event_id, mentions in event2mentions.items():
            event = Event(event_id, mentions)
            self.events.append(event)

    def to_ecr_data(self) -> EcrData:
        return EcrData(
            name=self.dataset_name,
            documents=self.documents,
            mentions=self.mentions,
            events=self.events,
            meta={"index_type": "word"},
        )
