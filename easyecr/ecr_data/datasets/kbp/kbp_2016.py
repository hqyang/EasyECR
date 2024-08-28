import re
import os
from typing import List
from typing import Dict
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element

from easyecr.ecr_data.data_structure.data_structure import Document
from easyecr.ecr_data.data_structure.data_structure import Mention
from easyecr.ecr_data.data_structure.data_structure import Event
from easyecr.ecr_data.data_structure.data_structure import EcrData
from easyecr.ecr_data.data_structure.data_structure import RichText
from easyecr.ecr_data.data_structure.data_structure import EcrData
from easyecr.ecr_data.datasets.kbp.source_reader import SourceReader
from easyecr.ecr_data.datasets.base_ecr_dataset.base_ecr_dataset import BaseEcrDataset

"""
Total number of ENG documents is (169), matching with golden 169;
Total number of ENG mentions is (4155), smaller than golden 4155;
Total number of ENG events is (3191), larger than golden 3191;
Dataset golden statistics from Github: https://github.com/jsksxs360/event-coref-emnlp2022

Total number of CMN documents is (167), matching with golden 167;
Total number of CMN mentions is (2518), smaller than golden 2518;
Total number of CMN events is (1912), larger than golden 1912;

Total number of SPA documents is (169), matching with golden 169;
Total number of SPA mentions is (2369), smaller than golden 2369;
Total number of SPA events is (1696), larger than golden 1696;
"""

KBPENG2016_TOTAL_NUM_OF_DOCUMENTS = 169
KBPENG2016_TOTAL_NUM_OF_MENTIONS = 4155
KBPENG2016_TOTAL_NUM_OF_EVENTS = 3191
KBPCMN2016_EVAL_NUM_OF_DOCUMENTS = 167
KBPCMN2016_EVAL_NUM_OF_MENTIONS = 2518
KBPCMN2016_EVAL_NUM_OF_EVENTS = 1912
KBPSPA2016_EVAL_NUM_OF_DOCUMENTS = 167
KBPSPA2016_EVAL_NUM_OF_MENTIONS = 2518
KBPSPA2016_EVAL_NUM_OF_EVENTS = 1912


class KBP2016Dataset(BaseEcrDataset):
    def __init__(self, dataset_name: str, directory: str):
        super().__init__(dataset_name, directory)
        self.documents = {}
        self.load_documents()
        self.mentions = {}
        self.events = []
        self.load_mentions_and_events()

    def load_documents(self):
        for dir in ["df", "nw"]:
            source_doc_dir = os.path.join(self.directory, f"{dir}/source")
            for doc_id, sentences in (
                SourceReader(self.dataset_name, source_doc_dir).read_source_folded_2016(dir == "df").items()
            ):
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

    def parse_extent(self, doc_id: str, trigger: str, offset: int, sents_list: Dict):
        return self.parse_helper(doc_id=doc_id, trigger=trigger, offset=offset, sents_list=sents_list, is_extent=True)

    def parse_anchor(self, doc_id: str, trigger: str, offset: int, sents_list: Dict):
        return self.parse_helper(doc_id=doc_id, trigger=trigger, offset=offset, sents_list=sents_list, is_extent=False)

    def parse_helper(self, doc_id: str, trigger: str, offset: int, sents_list: Dict, is_extent: bool):
        sent_list = sents_list[doc_id]
        trigger = re.sub(r"<.*?>", lambda x: " " * len(x.group()), trigger)
        for sent in sent_list:
            sent_start, sent_end = sent["start"], sent["start"] + len(sent["text"])
            if sent_start <= offset < sent_end:
                in_offset = offset - sent_start
                if any(map(lambda x: x in ["・", "."], ["・", "."])):  # some case
                    pass
                else:
                    assert sent["text"][in_offset : in_offset + len(trigger)] == trigger
                if is_extent:
                    return RichText(text=sent["text"], start=sent_start, end=sent_end - 1)
                else:
                    return RichText(text=trigger, start=in_offset, end=in_offset + len(trigger) - 1)

    def parse_argument(
        self,
        doc_id: str,
        arg_nodes: List[Element],
        entities_nodes: List[Element],
        sents_list: Dict,
    ):
        arguments = {}
        for arg_node in arg_nodes:
            text = arg_node.text.strip()
            role = arg_node.attrib["role"]
            if "entity_id" in arg_node.attrib:  # entity_arg
                entity_id = arg_node.attrib["entity_id"]
                entity_mention_id = arg_node.attrib["entity_mention_id"]
                for ent_node in entities_nodes:
                    if ent_node.attrib["id"] == entity_id:
                        for ent_mention_node in ent_node:
                            if ent_mention_node.attrib["id"] == entity_mention_id:
                                offset = int(ent_mention_node.attrib["offset"])
                                arguments[role] = self.parse_helper(
                                    doc_id=doc_id,
                                    trigger=text,
                                    offset=offset,
                                    sents_list=sents_list,
                                    is_extent=False,
                                )
        return arguments

    def parse_mentions_from_xml(self, doc_id: str, file_path: str, sents_list: Dict):
        tree = ET.ElementTree(file=file_path)
        entities = []
        for ent in tree.iter(tag="entity"):
            entities.append(ent)

        for hopper in tree.iter(tag="hopper"):
            h_id = hopper.attrib["id"]  # hopper id
            event_id = doc_id + "_" + h_id
            event_mentions = []
            for event in hopper.iter(tag="event_mention"):
                att = event.attrib
                mention_id = doc_id + "_" + att["id"]
                trigger_node = event.find("trigger")
                trigger = trigger_node.text.strip()
                offset = int(trigger_node.attrib["offset"])

                mention_extent = self.parse_extent(doc_id=doc_id, trigger=trigger, offset=offset, sents_list=sents_list)
                mention_anchor = self.parse_anchor(doc_id=doc_id, trigger=trigger, offset=offset, sents_list=sents_list)
                mention_anchor.start = mention_anchor.start + mention_extent.start
                mention_anchor.end = mention_anchor.end + mention_extent.start
                argument_nodes = event.findall("em_arg")
                mention_arguments = self.parse_argument(
                    doc_id=doc_id,
                    arg_nodes=argument_nodes,
                    entities_nodes=entities,
                    sents_list=sents_list,
                )
                meta = {
                    "mention_type": att["type"],
                    "mention_subtype": att["subtype"],
                    "mention_realis": att["realis"],
                    "original_mention_id": att["id"],
                    "event_id": event_id,
                }
                mention = Mention(
                    doc_id=doc_id,
                    mention_id=mention_id,
                    extent=mention_extent,
                    anchor=mention_anchor,
                    arguments=mention_arguments,
                    meta=meta,
                )
                self.mentions[mention_id] = mention
                event_mentions.append(mention)

            event = Event(event_id=event_id, mentions=event_mentions, meta=None)
            self.events.append(event)

    def load_mentions_and_events(self):
        for dir in ["df", "nw"]:
            source_doc_dir = os.path.join(self.directory, f"{dir}/source")
            sent_list = SourceReader(self.dataset_name, source_doc_dir).read_source_folded_2016(dir == "df")
            event_dir = os.path.join(self.directory, f"{dir}/ere")
            for file_name in os.listdir(event_dir):
                doc_id = re.sub(pattern="\.event_hoppers\.xml|\.rich_ere\.xml", repl="", string=file_name)
                self.parse_mentions_from_xml(
                    doc_id=doc_id, file_path=os.path.join(event_dir, file_name), sents_list=sent_list
                )

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
            meta={"index_type": "char"},
        )
