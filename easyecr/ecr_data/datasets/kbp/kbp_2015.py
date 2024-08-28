import re
import os
import xml.etree.ElementTree as ET
from typing import Union

from easyecr.ecr_data.data_structure.data_structure import Document
from easyecr.ecr_data.data_structure.data_structure import Mention
from easyecr.ecr_data.data_structure.data_structure import Event
from easyecr.ecr_data.data_structure.data_structure import EcrData
from easyecr.ecr_data.data_structure.data_structure import RichText
from easyecr.ecr_data.datasets.kbp.source_reader import SourceReader
from easyecr.ecr_data.datasets.base_ecr_dataset.base_ecr_dataset import BaseEcrDataset

"""
Total number of Eng documents is (158+202=360), matching with golden 360;
Total number of Eng mentions is (6538+6438=12976), matching with golden 12976;
Total number of Eng events is (3335+4125=7460),matching with golden 7460;

Dataset golden statistics from Paper: Joint Inference for Event Coreference Resolution
"""

KBPENG2015_TRAIN_NUM_OF_DOCUMENTS = 158
KBPENG2015_TRAIN_NUM_OF_MENTIONS = 6538
KBPENG2015_TRAIN_NUM_OF_EVENTS = 3335
KBPENG2015_EVAL_NUM_OF_DOCUMENTS = 202
KBPENG2015_EVAL_NUM_OF_MENTIONS = 6438
KBPENG2015_EVAL_NUM_OF_EVENTS = 4125


class KBPEng2015Dataset(BaseEcrDataset):
    def __init__(self, dataset_name: str, directory: str):
        super().__init__(dataset_name, directory)
        if "training" in directory:
            self.event_dir = os.path.join(directory, f"event_hopper")
        else:
            self.event_dir = os.path.join(directory, f"hopper")

        self.source_docs_dir = os.path.join(directory, f"source")
        self.source_reader = SourceReader(self.dataset_name, self.source_docs_dir)
        self.docid2sents = self.source_reader.read_source_folder_2015()

        self.documents = {}
        self.load_documents()
        self.mentions = {}
        self.events = []
        self.load_mentions_and_events()

    def load_documents(self):
        for doc_id, sentences in self.docid2sents.items():
            text = ""
            end = 0
            for sent in sentences:
                assert sent["start"] >= end
                text += " " * (sent["start"] - end)
                text += sent["text"]
                end = sent["start"] + len(sent["text"])

            for sent in sentences:  # check
                assert text[sent["start"] : sent["start"] + len(sent["text"])] == sent["text"]
            meta = None
            document = Document(doc_id=doc_id, text=text, meta=meta)
            self.documents[doc_id] = document

    def parse_extent_or_anchor(
        self, doc_id: str, trigger: str, trigger_start: int, is_extent: bool
    ) -> Union[RichText, None]:
        sent_list = self.docid2sents[doc_id]
        for sent in sent_list:
            sent_start, sent_end = sent["start"], sent["start"] + len(sent["text"])
            if sent_start <= trigger_start < sent_end:
                in_offset = trigger_start - sent_start
                assert sent["text"][in_offset : in_offset + len(trigger)] == trigger
                if is_extent:
                    return RichText(text=sent["text"], start=sent_start, end=sent_end - 1)
                else:
                    return RichText(text=trigger, start=in_offset, end=in_offset + len(trigger) - 1)
        return None

    def parse_mentions_from_xml(self, doc_id: str, file_path: str):
        tree = ET.ElementTree(file=file_path)
        for hopper in tree.iter(tag="hopper"):
            h_id = hopper.attrib["id"]  # hopper id
            event_id = doc_id + "_" + h_id
            event_mentions = []
            for event in hopper.iter(tag="event_mention"):
                att = event.attrib
                mention_id = doc_id + "_" + att["id"]
                trigger = event.find("trigger").text.strip()
                trigger_start = int(event.find("trigger").attrib["offset"])
                mention_extent = self.parse_extent_or_anchor(
                    doc_id=doc_id, trigger=trigger, trigger_start=trigger_start, is_extent=True
                )
                mention_anchor = self.parse_extent_or_anchor(
                    doc_id=doc_id, trigger=trigger, trigger_start=trigger_start, is_extent=False
                )
                mention_anchor.start = mention_anchor.start + mention_extent.start
                mention_anchor.end = mention_anchor.end + mention_extent.start
                meta = {
                    "mention_type": att["type"],
                    "mention_subtype": att["subtype"],
                    "mention_realis": att["realis"],
                    "original_mention_id": att["id"],
                }
                mention = Mention(
                    doc_id=doc_id,
                    mention_id=mention_id,
                    extent=mention_extent,
                    anchor=mention_anchor,
                    arguments=None,
                    meta=meta,
                )
                self.mentions[mention_id] = mention
                event_mentions.append(mention)

            event = Event(event_id=event_id, mentions=event_mentions, meta=None)
            self.events.append(event)

    def load_mentions_and_events(self):
        for file_name in os.listdir(self.event_dir):
            doc_id = re.sub(pattern="\.event_hoppers\.xml|\.rich_ere\.xml", repl="", string=file_name)
            self.parse_mentions_from_xml(doc_id=doc_id, file_path=os.path.join(self.event_dir, file_name))

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
