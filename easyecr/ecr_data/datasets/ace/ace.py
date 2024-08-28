import re
import glob
from bs4.element import Tag
from bs4 import BeautifulSoup

import spacy
from spacy.language import Language

from easyecr.ecr_data.data_structure.data_structure import Document
from easyecr.ecr_data.data_structure.data_structure import Mention
from easyecr.ecr_data.data_structure.data_structure import Event
from easyecr.ecr_data.data_structure.data_structure import EcrData
from easyecr.ecr_data.data_structure.data_structure import RichText
from easyecr.ecr_data.datasets.base_ecr_dataset.base_ecr_dataset import BaseEcrDataset

ACEChi2005_TOTAL_NUM_OF_DOCUMENTS = 633
ACEChi2005_TOTAL_NUM_OF_DOCUMENTS = 3333
ACEChi2005_TOTAL_NUM_OF_DOCUMENTS = 2521


class ACE2005Dataset(BaseEcrDataset):
    """
    ACE 2005 Multilingual Training Corpus https://catalog.ldc.upenn.edu/LDC2006T06
    """

    def __init__(self, dataset_name: str, directory: str):
        """

        :param dataset_name:
        :param directory:
        """
        super().__init__(dataset_name, directory)
        self.documents = {}
        self.load_documents()
        self.mentions = {}
        self.events = []
        self.load_events()

    def parse_target_node_text(self, node: Tag, target_node_name: str):
        """

        :param node:
        :param target_node_name:
        :return:
        """

        target_nodes = node.find_all(target_node_name)
        if target_nodes:
            result = target_nodes[0].text
        else:
            result = ""
        return result

    def parse_target_node_attr(self, node: Tag, target_node_name: str, attr: str):
        """

        :param node:
        :param target_node_name:
        :param attr:
        :return:
        """
        target_nodes = node.find_all(target_node_name)
        if target_nodes and attr in target_nodes[0].attrs:
            result = target_nodes[0][attr]
        else:
            result = ""
        return result

    def parse_doc_id_from_sgm(self, node: Tag):
        """

        :param node:
        :return:
        """
        result = self.parse_target_node_text(node, "DOCID")
        return result

    def parse_doc_type_from_sgm(self, node: Tag):
        """

        :param node:
        :return:
        """
        result = self.parse_target_node_text(node, "DOCTYPE")
        return result

    def parse_source_from_sgm(self, node: Tag):
        """

        :param node:
        :return:
        """
        result = self.parse_target_node_attr(node, "DOCTYPE", "SOURCE")
        return result

    def parse_text_from_sgm(self, file: str):
        """

        :param node:
        :return:
        """
        result = ""
        with open(file, encoding="utf-8", newline=None) as in_file:
            raw_text = in_file.read()
            result += re.sub(pattern=r"<[^>]*>", repl="", string=raw_text)
        return result

    def parse_datetime_from_sgm(self, node: Tag):
        """

        :param node:
        :return:
        """
        result = self.parse_target_node_text(node, "DATETIME")
        return result

    def parse_end_time_from_sgm(self, node: Tag):
        """

        :param node:
        :return:
        """
        result = self.parse_target_node_text(node, "ENDTIME")
        return result

    def parse_headline_from_sgm(self, node: Tag):
        """

        :param node:
        :return:
        """
        result = self.parse_target_node_text(node, "HEADLINE")
        return result

    def load_documents(self):
        """

        :return:
        """
        if "English" in self.directory:
            model_name = "en_core_web_sm"
        elif "Chinese" in self.directory:
            model_name = "zh_core_web_sm"
        else:
            raise NotImplementedError

        lang_model = spacy.load(model_name)

        sgm_files = glob.glob(self.directory + "/**/adj/*.sgm", recursive=True)
        for file in sgm_files:
            content = self.read_all_content(file)
            root_node = BeautifulSoup(content, features="xml")
            doc_id = self.parse_doc_id_from_sgm(root_node).strip()
            doc_type = self.parse_doc_type_from_sgm(root_node)
            doc_source = self.parse_source_from_sgm(root_node)
            doc_text = self.parse_text_from_sgm(file)
            document_datetime = self.parse_datetime_from_sgm(root_node)
            document_end_time = self.parse_end_time_from_sgm(root_node)
            headline = self.parse_headline_from_sgm(root_node)
            meta = {
                "doc_type": doc_type,
                "doc_source": doc_source,
                "document_datetime": document_datetime,
                "document_end_time": document_end_time,
                "headline": headline,
            }
            doc = Document(doc_id, doc_text, meta)
            self.documents[doc_id] = doc

    def parse_doc_id_from_apf(self, node: Tag):
        """

        :param node:
        :return:
        """
        result = self.parse_target_node_attr(node, "document", "DOCID")
        return result

    def parse_charseq_node(self, node: Tag):
        """

        :param charseq:
        :return:
        """
        charseq_node = node.find("charseq")
        result = RichText(charseq_node.text, int(charseq_node["START"]), int(charseq_node["END"]))
        return result

    def parse_mention_extent(self, event_mention_node: Tag):
        """

        :param event_mention_node:
        :return:
        """
        extent_node = event_mention_node.find("extent", recursive=False)
        result = self.parse_charseq_node(extent_node)
        return result

    def parse_mention_anchor(self, event_mention_node: Tag):
        """

        :param event_mention_node:
        :return:
        """
        anchor_node = event_mention_node.find("anchor", recursive=False)
        result = self.parse_charseq_node(anchor_node)
        return result

    def parse_single_argument(self, argument_node: Tag):
        """

        :param argument_node:
        :return:
        """
        role = argument_node["ROLE"]
        extent_node = argument_node.find("extent")
        text = self.parse_charseq_node(extent_node)
        result = {"role": role, "text": text}
        return result

    def parse_mention_arguments(self, event_mention_node: Tag):
        """

        :param event_mention_node:
        :return:
        """
        event_mention_argument_nodes = event_mention_node.find_all("event_mention_argument")
        result = []
        for argument_node in event_mention_argument_nodes:
            argument = self.parse_single_argument(argument_node)
            result.append(argument)
        return result

    def parse_event_mention(self, event_mention_node: Tag, doc_id: str, event_id: str):
        """

        :param event_mention_node:
        :param doc_id:
        :return:
        """
        mention_id = event_mention_node["ID"]
        mention_extent = self.parse_mention_extent(event_mention_node)
        mention_anchor = self.parse_mention_anchor(event_mention_node)
        mention_arguments = self.parse_mention_arguments(event_mention_node)
        meta = {"event_id": event_id}
        result = Mention(doc_id, mention_id, mention_extent, mention_anchor, mention_arguments, meta)
        self.mentions[mention_id] = result
        return result

    def parse_single_event(self, event_node: Tag, doc_id: str):
        """

        :param event_node:
        :param doc_id:
        :return:
        """
        event_id = event_node.attrs["ID"]
        mentions = []
        event_mention_nodes = event_node.find_all("event_mention")
        for event_mention_node in event_mention_nodes:
            mention = self.parse_event_mention(event_mention_node, doc_id, event_id)
            mention.add_tag(name="subtype", value=event_node["SUBTYPE"])
            mentions.append(mention)
        meta = {
            "type": event_node["TYPE"],
            "subtype": event_node["SUBTYPE"],
            "modality": event_node["MODALITY"],
            "polarity": event_node["POLARITY"],
            "genericity": event_node["GENERICITY"],
            "tense": event_node["TENSE"],
        }
        result = Event(event_id, mentions, meta)
        self.events.append(result)

    def parse_events(self, node: Tag, doc_id: str):
        """

        :param node:
        :param doc_id:
        :return:
        """
        event_nodes = node.find_all("event")
        for event_node in event_nodes:
            self.parse_single_event(event_node, doc_id)

    def load_events(self):
        """

        :return:
        """
        apf_files = glob.glob(self.directory + "/**/adj/*.apf.xml", recursive=True)
        for file in apf_files:
            content = self.read_all_content(file)
            root_node = BeautifulSoup(content, features="xml")
            doc_id = self.parse_doc_id_from_apf(root_node)
            self.parse_events(root_node, doc_id)

    def to_ecr_data(self, verbose: bool = False) -> "EcrData":
        """

        :return:
        """
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
