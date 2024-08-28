#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""



Date: 2023/11/20 10:00
"""
import os
import csv
from typing import List
from collections import defaultdict

from easyecr.ecr_data.data_structure.data_structure import Document
from easyecr.ecr_data.data_structure.data_structure import Mention
from easyecr.ecr_data.data_structure.data_structure import Event
from easyecr.ecr_data.data_structure.data_structure import EcrData
from easyecr.ecr_data.data_structure.data_structure import RichText
from easyecr.ecr_data.datasets.base_ecr_dataset.base_ecr_dataset import BaseEcrDataset


"""
all
Total number of documents is (358+78+74=510), matching with golden 510;
Total number of mentions is (5313+977+1008=7298), matching with golden 7298;
Total number of events is (991+228+194=1413), larger than golden 1411;
train
Number of train documents is 358, matching with golden 358;
Number of train mentions is 5313, matching with golden 5313;
Number of train events is 991, matching with golden 991;
dev
Number of dev documents is 78, matching with golden 78;
Number of dev mentions is 977, matching with golden 977;
Number of dev events is 228, matching with golden 228;
test
Number of test documents is 74, matching with golden 74;
Number of test mentions is 1008, matching with golden 1008;
Number of test events is 194, matching with golden 194;

Dataset golden statistics from Paper: Don’t Annotate, but Validate: a Data-to-Text Method for Capturing Event Data (Vossen et al., LREC 2018)
Data set partition reference Github: https://github.com/UKPLab/cdcr-beyond-corpus-tailored/tree/master/resources/data/gun_violence
"""
GVC_TOTAL_NUM_OF_DOCUMENTS = 510
GVC_TOTAL_NUM_OF_MENTIONS = 7298
GVC_TOTAL_NUM_OF_EVENTS = 1411
GVC_TRAIN_NUM_OF_DOCUMENTS = 358
GVC_TRAIN_NUM_OF_MENTIONS = 5313
GVC_TRAIN_NUM_OF_EVENTS = 991
GVC_DEV_NUM_OF_DOCUMENTS = 78
GVC_DEV_NUM_OF_MENTIONS = 977
GVC_DEV_NUM_OF_EVENTS = 228
GVC_TEST_NUM_OF_DOCUMENTS = 74
GVC_TEST_NUM_OF_MENTIONS = 1008
GVC_TEST_NUM_OF_EVENTS = 194


class GVCDataset(BaseEcrDataset):
    def __init__(self, dataset_name: str = "gvc", directory: str = "~/dataset/"):
        """_summary_

        Args:
            dataset_name (str): Default 'gvc'
            directory (str): Directory stored dataset, the parent directory of file 'gold.conll'
        """
        super().__init__(dataset_name, directory)
        self.documents = {}
        self.load_documents()
        self.mentions = {}
        self.load_mentions()
        self.events = []
        self.load_events()

    def _open_file(self) -> List[str]:
        """Open a file, and return a stream

        Returns:
            list[str]: A list of lines
        """
        with open(self.directory, "r") as file:
            return file.readlines()

    def parse_subtopic(self, doc_id: str):
        base_dir = os.path.dirname(self.directory)
        with open(os.path.join(base_dir, "gvc_split/gvc_doc_to_event.csv"), newline="") as cvs_file:
            reader = csv.reader(cvs_file)
            next(reader)
            for row in reader:
                if row[0] == doc_id:
                    return str(row[1])
        return ""

    def load_documents(self):
        """Each document is processed as a Document instance and stored in self.documents according to the doc_id."""
        raw_data = self._open_file()
        doc_title = []
        doc_info = []
        doc_pub_time = None
        global_token_idx = 0
        for line in raw_data:
            if line.startswith("#begin"):
                doc_id = line.strip().split()[2][1:-2]
                doc_subtopic = self.parse_subtopic(doc_id=doc_id)
            elif line.startswith("#end"):
                doc_token = doc_info
                doc_text = []
                for item in doc_info:
                    doc_text.append(item["text"])
                doc_text = " ".join(doc_text)
                meta = {
                    "doc_title": " ".join(doc_title),
                    "doc_pub_time": doc_pub_time,
                    "doc_token": doc_token,
                    "doc_topic": "1",
                    "doc_subtopic": doc_subtopic,
                }
                self.documents[doc_id] = Document(doc_id=doc_id, text=doc_text, meta=meta)
                doc_title = []
                doc_info = []
                global_token_idx = 0
                doc_pub_time = None
            else:
                (
                    token_idx_conflated,
                    token,
                    sentence_type,
                    _,
                ) = line.strip().split("\t")

                # detect special lines with publish date information
                if sentence_type == "DCT":
                    doc_pub_time = token
                    continue

                if token == "NEWLINE" or not token.strip():
                    continue

                if sentence_type == "TITLE":
                    doc_title.append(token)
                # doc_text.append(token)
                _, sentence_type_and_idx, token_idx = token_idx_conflated.split(".")
                sent_idx, number = sentence_type_and_idx, token_idx
                if sent_idx == "t1":
                    sent_idx = "0"
                    number = int(number) - 1
                elif sent_idx == "b1":
                    sent_idx = "1"
                    number = int(number) - 1
                else:
                    sent_idx = sentence_type_and_idx[1:]
                    number = int(number)

                doc_info.append({"sentence": sent_idx, "number": number, "text": token, "t_id": global_token_idx})
                global_token_idx += 1

    def parse_mentions_from_document(self, doc_id: str, doc_sentence_info: defaultdict, doc_mention_info: defaultdict):
        doc_mentions = {}
        for m_id, m_info in doc_mention_info.items():
            sen_idx = m_info["sen_idx"]
            event_id = m_info["event_id"]

            token_idx, token = zip(*doc_sentence_info[sen_idx])
            # Extent
            extent_words = list(token)
            extent_start = list(token_idx)[0]
            extent_end = list(token_idx)[-1]
            extent_text = " ".join(extent_words)
            extent = RichText(extent_text, int(extent_start), int(extent_end) - 1, extent_words)
            # Anchor location information (sentence level and document level)
            sen_loc_start = m_info["sen_loc_start"]
            sen_loc_end = m_info["sen_loc_end"]
            doc_loc_start = m_info["doc_loc_start"]
            doc_loc_end = m_info["doc_loc_end"]
            # Anchor
            anchor_words = [extent_words[i] for i in range(sen_loc_start, sen_loc_end)]
            anchor_start = doc_loc_start
            anchor_end = doc_loc_end
            anchor_text = " ".join(anchor_words)
            anchor = RichText(anchor_text, int(anchor_start), int(anchor_end) - 1, anchor_words)
            # Meta
            meta = {"event_id": event_id, "sentence_idx": sen_idx}
            mention = Mention(doc_id, m_id, extent, anchor, arguments=None, meta=meta)
            doc_mentions[m_id] = mention
        return doc_mentions

    def load_mentions(self):
        raw_data = self._open_file()
        # Unique in the corpus
        mention_id = -1
        #
        doc_token_idx = -1  # Increment within document
        doc_sentence_info = defaultdict(list)  # Sentence_idx map to words of sentence
        doc_mention_info = defaultdict(dict)  # Mention_id map to mention_info
        #
        sen_token_idx = -1  # Increment within sentence
        #
        sen_mention_anchor_start, sen_mention_anchor_end = None, None
        doc_mention_anchor_start, doc_mention_anchor_end = None, None

        def mention_helper(
            token_idx_conflated,
            doc_sen_idx,
            label_text,
            sen_token_idx,
            doc_token_idx,
            doc_mention_info,
        ):
            """Auxiliary statistics related information of each mention

            Args:
                doc_sen_idx (_type_): Index of sentences within a document
                label_text (_type_): The event cluster to which mention belongs
                sen_token_idx (_type_): Index of tokens within a sentence
                doc_token_idx (_type_): Index of tokens within a document
                doc_mention_info (_type_): A dictionary that records mention information in a document
            """
            event_id = int(label_text.replace("(", "").replace(")", ""))

            nonlocal mention_id
            nonlocal sen_mention_anchor_start
            nonlocal sen_mention_anchor_end
            nonlocal doc_mention_anchor_start
            nonlocal doc_mention_anchor_end

            if "(" in label_text:
                mention_id = token_idx_conflated  # update mention_id
                sen_mention_anchor_start = sen_token_idx
                doc_mention_anchor_start = doc_token_idx
            if ")" in label_text:
                sen_mention_anchor_end = sen_token_idx + 1
                doc_mention_anchor_end = doc_token_idx + 1
            doc_mention_info[mention_id]["sen_idx"] = doc_sen_idx
            doc_mention_info[mention_id]["event_id"] = event_id

            check_not_none = lambda x: x is not None  #

            if all(
                map(
                    check_not_none,
                    [
                        sen_mention_anchor_start,
                        sen_mention_anchor_end,
                        doc_mention_anchor_start,
                        doc_mention_anchor_end,
                    ],
                )
            ):
                doc_mention_info[mention_id]["sen_loc_start"] = sen_mention_anchor_start
                doc_mention_info[mention_id]["sen_loc_end"] = sen_mention_anchor_end
                doc_mention_info[mention_id]["doc_loc_start"] = doc_mention_anchor_start
                doc_mention_info[mention_id]["doc_loc_end"] = doc_mention_anchor_end
                sen_mention_anchor_start, sen_mention_anchor_end = None, None
                doc_mention_anchor_start, doc_mention_anchor_end = None, None
            elif all(map(check_not_none, [sen_mention_anchor_start, doc_mention_anchor_start])):
                doc_mention_info[mention_id]["sen_loc_start"] = sen_mention_anchor_start
                doc_mention_info[mention_id]["doc_loc_start"] = doc_mention_anchor_start
            elif all(map(check_not_none, [sen_mention_anchor_end, doc_mention_anchor_end])):
                doc_mention_info[mention_id]["sen_loc_end"] = sen_mention_anchor_end
                doc_mention_info[mention_id]["doc_loc_end"] = doc_mention_anchor_end
            else:
                pass

        for line in raw_data:
            if line.startswith("#begin"):
                doc_id = line.strip().split()[2][1:-2]
            elif line.startswith("#end"):
                doc_mentions = self.parse_mentions_from_document(doc_id, doc_sentence_info, doc_mention_info)
                self.mentions.update(doc_mentions)
                # reset variable of doc
                doc_token_idx = -1
                doc_sentence_info = defaultdict(list)
                doc_mention_info = defaultdict(dict)
            else:
                (
                    token_idx_conflated,
                    token,
                    sentence_type,
                    label_text,
                ) = line.strip().split("\t")

                if sentence_type == "DCT" or token == "NEWLINE" or not token.strip():
                    continue

                doc_token_idx += 1

                _, sentence_type_and_idx, token_idx = token_idx_conflated.split(".")
                if (sentence_type_and_idx in ["t1", "b1"] and token_idx == "1") or (
                    sentence_type_and_idx not in ["t1", "b1"] and token_idx == "0"
                ):  # The raw token index of sentence in t1 and b1 start with 1, others start with 0.
                    sen_token_idx = -1  # new sentence, reset
                sen_token_idx += 1
                if sentence_type_and_idx == "t1":
                    doc_sen_idx = "0"
                else:
                    doc_sen_idx = sentence_type_and_idx[1:]
                doc_sentence_info[doc_sen_idx].append((doc_token_idx, token))
                if not label_text == "-":
                    mention_helper(
                        token_idx_conflated,
                        doc_sen_idx,
                        label_text,
                        sen_token_idx,
                        doc_token_idx,
                        doc_mention_info,
                    )

    def load_events(self):
        """Merge the mentions into an event based on the mention_id"""
        event2mentions = defaultdict(list)
        for _, mention in self.mentions.items():
            event_id = mention.meta["event_id"]
            event2mentions[event_id].append(mention)
        for event_id, mentions in event2mentions.items():
            event = Event(event_id, mentions)
            self.events.append(event)

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
