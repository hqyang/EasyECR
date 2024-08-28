import os
import json
import hashlib
from typing import List
from typing import Dict

from easyecr.ecr_data.data_structure.data_structure import Document
from easyecr.ecr_data.data_structure.data_structure import Mention
from easyecr.ecr_data.data_structure.data_structure import Event
from easyecr.ecr_data.data_structure.data_structure import EcrData
from easyecr.ecr_data.data_structure.data_structure import RichText
from easyecr.ecr_data.datasets.base_ecr_dataset.base_ecr_dataset import BaseEcrDataset


class IDEACarDataset(BaseEcrDataset):
    def __init__(self, dataset_name: str, directory: str):
        super().__init__(dataset_name, directory)
        self.documents = {}
        self.load_documents()
        self.mentions = {}
        self.events = []
        self.load_events()

    def convert_doc_desc_to_id(self, input_string: str):
        hash_object = hashlib.sha256()
        hash_object.update(input_string.encode())
        hash_hex = hash_object.hexdigest()
        int_value = int(hash_hex[:8], 16)
        return int_value

    def load_raw_data_from_json(self, file_path: str):
        with open(file_path, "r") as f:
            data = json.load(f)
        return data

    def parse_per_document_of_event(self, event_desc: str, event_docs: List[Dict]):
        for doc_info in event_docs:
            doc_id = doc_info["original_data"]["id"]
            text = doc_info["original_data"]["event"]
            meta = {
                "url": doc_info["original_data"]["url"],
                "pub_date": doc_info["original_data"]["发表时间"],
            }
            self.documents[doc_id] = Document(doc_id, text, meta)

    def load_documents(self):
        for file_name in os.listdir(self.directory):
            file_path = os.path.join(self.directory, file_name)
            data = self.load_raw_data_from_json(file_path)
            for event_desc, event_docs in data.items():
                self.parse_per_document_of_event(event_desc, event_docs)

    def parse_argument_from_document(self, doc_info: Dict):
        return doc_info["original_annotation"]["mention_info"]

    def parse_extent_from_document(self, doc_info: Dict, arguments: Dict):
        locs = set()
        for key, value in arguments.items():
            if key.endswith("start") or key.endswith("end"):
                locs.update({value})
        start = min(locs)
        end = max(locs)
        text = doc_info["original_data"]["event"][start:end]
        return RichText(text, start, end - 1)

    def parse_anchor_from_document(self, arguments: Dict):
        text, start, end = None, None, None
        if "谓语" in arguments and arguments["谓语"]:
            text = arguments["谓语"]
            start = arguments["谓语_start"]
            end = arguments["谓语_end"]
        elif "涉事主体" in arguments and arguments["涉事主体"]:
            text = arguments["涉事主体"]
            start = arguments["涉事主体_start"]
            end = arguments["涉事主体_end"]
        elif "宾语" in arguments and arguments["宾语"]:
            text = arguments["宾语"]
            start = arguments["宾语_start"]
            end = arguments["宾语_end"]
        else:
            pass
        return RichText(text, start, end - 1)

    def parse_mention_from_document(self, idx, doc_info: Dict):
        doc_id = doc_info["original_data"]["id"]
        mention_id = f"m_{doc_id}_{idx}"
        arguments = self.parse_argument_from_document(doc_info)
        extent = self.parse_extent_from_document(doc_info, arguments)
        anchor = self.parse_anchor_from_document(arguments)
        meta = {
            "annotation_template_str": doc_info["original_annotation"]["annotation_template_str"],
            "event_type": doc_info["original_annotation"]["event_type"],
        }

        mention = Mention(doc_id, mention_id, extent, anchor, arguments, meta)
        self.mentions[mention_id] = mention
        return mention

    def parse_per_mention_of_event(self, event_desc: str, event_docs: List[Dict]):
        event_id = self.convert_doc_desc_to_id(event_desc)
        mentions = []
        event_type = set()
        for idx, doc_info in enumerate(event_docs):
            doc_id = doc_info["original_data"]["id"]
            event_type.update({doc_info["event_type"]})
            mention = self.parse_mention_from_document(idx, doc_info)
            mentions.append(mention)
            meta = {
                "event_desc": event_desc,
                "type": list(event_type)[0],
                "doc_id": doc_id,
            }
        self.events.append(Event(event_id, mentions, meta))

    def load_events(self):
        for file_name in os.listdir(self.directory):
            file_path = os.path.join(self.directory, file_name)
            data = self.load_raw_data_from_json(file_path)
            for event_desc, event_docs in data.items():
                self.parse_per_mention_of_event(event_desc, event_docs)

    def to_ecr_data(self) -> EcrData:
        return EcrData(
            name=self.dataset_name,
            documents=self.documents,
            mentions=self.mentions,
            events=self.events,
            meta={"index_type": "char"},
        )
