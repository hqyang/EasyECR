import os
import fnmatch
from typing import Dict
from typing import List
from collections import defaultdict

import numpy as np
from bs4 import ResultSet
from bs4.element import Tag
from bs4 import BeautifulSoup
from bs4 import NavigableString

from easyecr.ecr_data.datasets.base_ecr_dataset.base_ecr_dataset import BaseEcrDataset
from easyecr.ecr_data.data_structure.data_structure import Document
from easyecr.ecr_data.data_structure.data_structure import Mention
from easyecr.ecr_data.data_structure.data_structure import Event
from easyecr.ecr_data.data_structure.data_structure import EcrData
from easyecr.ecr_data.data_structure.data_structure import RichText

"""
all
Total number of documents is (574+196+206=976), matching with golden 976;
Total number of mentions is (3808+1245+1780=6833), matching with golden 6833;
Total number of events is (1527+409+805=2741), matching with golden 2741;
train
Number of train documents is 574, matching with golden 574;
Number of train mentions is 3808, matching with golden 3808;
Number of train events is 1527, matching with golden 1527;
dev
Number of dev documents is 196, matching with golden 196;
Number of dev mentions is 1245, matching with golden 1245;
Number of dev events is 409, matching with golden 409;
test
Number of test documents is 206, matching with golden 206;
Number of test mentions is 1780, matching with golden 1780;
Number of test events is 805, matching with golden 805;

Dataset golden statistics from Paper: Contrastive Representation Learning for Cross-Document Coreference Resolution of Events and Entities
"""


class ECBPlusDataset(BaseEcrDataset):
    """
    2014-lrec-Using a sledgehammer to crack a nut? Lexical diversity and event coreference resolution [paper](./paper/2014-lrec-Using%20a%20sledgehammer%20to%20crack%20a%20nut%20Lexical%20diversity%20and%20event%20coreference%20resolution.pdf) [github](https://github.com/cltl/ecbPlus.git)
    """

    def __init__(self, dataset_name: str, directory: str):
        """

        :param dataset_name:
        :param directory:
        """
        super().__init__(dataset_name, directory)
        self.singleton_id = 674242636155
        self.uni_mention_id = -1  # Unique within the corpus
        self.documents = {}
        self.load_documents()
        self.mentions = {}
        self.load_mentions()
        self.events = []
        self.load_events()

    @property
    def annotated_sentences(self):
        validated_sentences = np.genfromtxt(
            os.path.join(self.directory, "ECBplus_coreference_sentences.csv"),
            delimiter=",",
            dtype=str,
            skip_header=1,
        )
        sentences = {}
        for topic, doc, sentence in validated_sentences:
            if topic not in sentences:
                sentences[topic] = {}
            doc_name = topic + "_" + doc + ".xml"
            if doc_name not in sentences[topic]:
                sentences[topic][doc_name] = []
            sentences[topic][doc_name].append(sentence)
        return sentences

    def parse_doc_id_from_xml(self, node: Tag):
        return node.find("document")["doc_id"]

    def parse_text_from_xml(self, node: Tag):
        token_text_list = [sub_node.text for sub_node in node.find_all("token")]
        return " ".join(token_text_list)

    def parse_token_from_xml(self, node: Tag):
        token_meta_list = [
            {
                "t_id": sub_node.attrs["t_id"],
                "sentence": sub_node.attrs["sentence"],
                "number": sub_node.attrs["number"],
                "text": sub_node.text,
            }
            for sub_node in node.find_all("token")
        ]
        return token_meta_list

    def load_per_topic_documents(self, topic_path: str, topic_validated_doc: Dict):
        pattern = "*xml"
        for doc_name in os.listdir(topic_path):
            if fnmatch.fnmatch(doc_name, pattern) and doc_name in topic_validated_doc:
                doc_path = os.path.join(topic_path, doc_name)
                content = self.read_all_content(doc_path)
                root_node = BeautifulSoup(content, "html.parser")
                doc_id = self.parse_doc_id_from_xml(root_node)
                doc_text = self.parse_text_from_xml(root_node)
                doc_token = self.parse_token_from_xml(root_node)
                doc_topic = doc_name.split("_")[0]
                doc_subtopic = "0" if "plus" in doc_name else "1"
                meta = {
                    "doc_name": doc_name,
                    "doc_topic": doc_topic,
                    "doc_subtopic": f"{doc_topic}_{doc_subtopic}",
                    "doc_token": doc_token,
                    "doc_sentences": topic_validated_doc[doc_name],
                }
                self.documents[doc_id] = Document(doc_id, doc_text, meta)

    def load_documents(self):
        validated_sentences = self.annotated_sentences
        data_dir = os.path.join(self.directory, "ECB+")
        for topic in os.listdir(data_dir):
            topic_path = os.path.join(data_dir, topic)
            if os.path.isdir(topic_path):
                self.load_per_topic_documents(topic_path, validated_sentences[topic])

    def parse_mention_extent(self, mention_token_ids: list, token_nodes: ResultSet):
        mention_sentence_ids = set()
        for t_id in mention_token_ids:
            node = [token_node for token_node in token_nodes if token_node["t_id"] == t_id][0]
            # mention_sentence_ids.update(node["sentence"])
            mention_sentence_ids.add(node["sentence"])

        mention_tokens = []
        sentence_token_ids = []
        for sentence_id in mention_sentence_ids:
            mention_tokens.extend([node.text for node in token_nodes if node["sentence"] == sentence_id])
            sentence_token_ids.extend([node["t_id"] for node in token_nodes if node["sentence"] == sentence_id])
        text = " ".join(mention_tokens)
        start = int(sentence_token_ids[0]) - 1
        end = int(sentence_token_ids[-1]) - 1
        words = mention_tokens
        return RichText(text=text, start=start, end=end, words=words)

    def parse_mention_anchor(self, mention_token_ids: list, token_nodes: ResultSet):
        mention_text = []
        for t_id in mention_token_ids:
            mention_text.extend([node.text for node in token_nodes if node["t_id"] == t_id])
        return RichText(
            text=" ".join(mention_text),
            start=int(mention_token_ids[0]) - 1,
            end=int(mention_token_ids[-1]) - 1,
            words=mention_text,
        )

    def parse_mention_argument(self):
        return None

    def parse_mention_sentenceid(self, mention_token_ids: list, token_nodes: ResultSet):
        s_ids = []
        for t_id in mention_token_ids:
            s_ids.extend([node["sentence"] for node in token_nodes if node["t_id"] == t_id])
        return int(s_ids[0])

    def parse_mention_from_doc(
        self,
        doc_id,
        doc_name,
        mention_nodes,
        validated_sentences,
        token_nodes,
    ):
        mentions = {}
        for mention in mention_nodes:
            if mention.name.startswith("act") or mention.name.startswith("neg"):  # EVENT
                token_ids = []
                for token in mention.contents:
                    if isinstance(token, NavigableString):
                        continue
                    token_ids.append(token["t_id"])

                sentence_id = self.parse_mention_sentenceid(token_ids, token_nodes)
                if sentence_id not in validated_sentences:  # whether is annotated sentence
                    continue
                anchor = self.parse_mention_anchor(token_ids, token_nodes)
                mention_id = mention["m_id"]
                extent = self.parse_mention_extent(token_ids, token_nodes)
                arguments = self.parse_mention_argument()
                meta = {"doc_name": doc_name, "sentence_id": sentence_id}
                mention = Mention(
                    doc_id=doc_id,
                    mention_id=mention_id,
                    extent=extent,
                    anchor=anchor,
                    arguments=arguments,
                    meta=meta,
                )
                mentions[mention_id] = mention
            else:  # NO EVENT
                pass
        return mentions

    def parse_event_mention_from_doc(self, node, event_nodes, mentions):
        mention_cluster_info = {}
        for event_node in event_nodes:
            mention_cluster_info[event_node["m_id"]] = {
                "event_id": event_node.attrs.get("instance_id", ""),
                "event_desc": event_node["tag_descriptor"],
            }

        relation_tag, relation_rid, relation_source_target = {}, {}, {}
        relation_nodes = node.find_all(attrs={"r_id": True})
        for relation in relation_nodes:
            target_mention = relation.contents[-2]["m_id"]
            relation_tag[target_mention] = relation.name
            relation_rid[target_mention] = relation["r_id"]
            for mention in relation:
                if mention.name == "source":
                    relation_source_target[mention["m_id"]] = target_mention

        for m_id, mention in mentions.items():
            meta = mention.meta
            target = relation_source_target.get(m_id, None)
            if target is None:  # singleton mention
                self.singleton_id += 1
                meta["event_id"] = self.singleton_id
                meta["event_desc"] = ""
            else:  # no singleton mention
                r_id = relation_rid[target]
                tag = relation_tag[target]
                if tag.startswith("intra"):
                    event_id = int(r_id + "1")
                else:
                    event_id = int(mention_cluster_info[target]["event_id"][3:])
                meta["event_id"] = event_id
                meta["event_desc"] = mention_cluster_info[target]["event_desc"]
            mention.meta = meta
            self.uni_mention_id += 1  # update mention_id

            mention.add_tag("original_mention_id", mention.mention_id)
            mention.mention_id = str(self.uni_mention_id)
            self.mentions[mention.mention_id] = mention

    def parse_mentions_from_doc(
        self,
        node: Tag,
        doc_id: str,
        doc_name: str,
        validated_sentences: List,
    ):
        token_nodes = node.find_all("token")
        markable_nodes = node.find("markables").contents
        event_nodes, mention_nodes = [], []
        for markable_node in markable_nodes:
            if isinstance(markable_node, NavigableString):
                continue
            if "related_to" in markable_node.attrs:
                event_nodes.append(markable_node)
            else:
                mention_nodes.append(markable_node)

        mentions = self.parse_mention_from_doc(
            doc_id,
            doc_name,
            mention_nodes,
            validated_sentences,
            token_nodes,
        )
        # Recall event information for mention
        self.parse_event_mention_from_doc(node, event_nodes, mentions)

    def load_per_topic_mentions(self, topic_path: str, topic_validated_docs: Dict):
        pattern = "*xml"
        for doc_name in os.listdir(topic_path):
            if fnmatch.fnmatch(doc_name, pattern) and doc_name in topic_validated_docs:
                doc_path = os.path.join(topic_path, doc_name)
                content = self.read_all_content(doc_path)
                root_node = BeautifulSoup(content, "html.parser")
                doc_id = self.parse_doc_id_from_xml(root_node)
                self.parse_mentions_from_doc(
                    root_node,
                    doc_id,
                    doc_name,
                    sorted(list(map(int, topic_validated_docs[doc_name]))),
                )

    def load_events(self):
        event2mentions = defaultdict(list)
        for _, mention in self.mentions.items():
            event_id = mention.meta["event_id"]
            event2mentions[event_id].append(mention)
        for event_id, mentions in event2mentions.items():
            event = Event(event_id, mentions)
            self.events.append(event)

    def load_mentions(self):
        validated_sentences = self.annotated_sentences
        data_dir = os.path.join(self.directory, "ECB+")
        for topic in os.listdir(data_dir):
            topic_path = os.path.join(data_dir, topic)
            if os.path.isdir(topic_path):
                self.load_per_topic_mentions(topic_path, validated_sentences[topic])

    def to_ecr_data(self, verbose: bool = False) -> "EcrData":
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
            meta={"index_type": "word"},
        )
