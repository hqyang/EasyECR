import json
from typing import List
from collections import defaultdict

import spacy
from spacy.tokens import Doc
from spacy.language import Language

from easyecr.ecr_data.data_structure.data_structure import Document
from easyecr.ecr_data.data_structure.data_structure import Mention
from easyecr.ecr_data.data_structure.data_structure import Event
from easyecr.ecr_data.data_structure.data_structure import EcrData
from easyecr.ecr_data.data_structure.data_structure import RichText
from easyecr.ecr_data.datasets.base_ecr_dataset.base_ecr_dataset import BaseEcrDataset


"""
Total number of Eng mentions is (40529+1250+1893=43672), matching with golden 43672;
Total number of Eng events is (7042+233+322=7597), matching with golden 7597;

Paper Ref: WEC: Deriving a Large-scale Cross-document Event Coreference dataset from Wikipedia (Eirew et al., NAACL 2021)
Statistical constant of WEC-Eng
"""
WECENG_TRAIN_NUM_OF_MENTIONS = 40529
WECENG_TRAIN_NUM_OF_EVENTS = 7042
WECENG_DEV_NUM_OF_MENTIONS = 1250
WECENG_DEV_NUM_OF_EVENTS = 233
WECENG_TEST_NUM_OF_MENTIONS = 1893
WECENG_TEST_NUM_OF_EVENTS = 322


class WECEngDataset(BaseEcrDataset):
    def __init__(
        self,
        dataset_name: str = "wec-eng",
        directory: str = "~/dataset/",
    ):
        """_summary_

        Args:
            dataset_name (str): Default 'wec-eng'
            directory (str): Directory stored dataset, the parent directory of file '*.json'
            selection (str, optional): Support Train, Dev, Test set. Defaults to "Train".
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
            List[str]: A list of lines
        """
        with open(self.directory, "r") as file:
            return json.load(file)

    def parse_doc_token(self, context: List, lang_model: Language):
        spaces = [True for _ in context]
        spaces[-1] = False
        raw_doc = Doc(lang_model.vocab, words=context, spaces=spaces)
        doc = lang_model(raw_doc)
        doc_token = []
        global_token_idx = 0
        for i, sen in enumerate(doc.sents):
            for j, token in enumerate(sen):
                doc_token.append({"t_id": global_token_idx, "sentence": i, "number": j, "text": token.text})
                global_token_idx += 1
        return doc_token

    def load_documents(self):
        """Each document is processed as a Document instance and stored in self.documents according to the doc_id."""
        model_name = "en_core_web_sm"
        lang_model = spacy.load(model_name, exclude=["ner"])
        raw_data = self._open_file()
        for mention_info in raw_data:
            doc_id = mention_info["doc_id"]
            context = mention_info["mention_context"]
            text = " ".join(context)
            doc_token = self.parse_doc_token(context=context, lang_model=lang_model)
            meta = {"doc_token": doc_token}
            document = Document(doc_id, text, meta)
            self.documents[doc_id] = document

    def parse_mention_anchor(self, tokens_str: str, tokens_idx: List[int], mention_head: str, doc: Doc) -> RichText:
        """Processing mention anchor as a RichText instance

        Args:
            tokens_str (str): Mention span text
            tokens_idx (list[int]): Index of 'mention span text' within a document
            mention_head (str): Mention span head token
            doc (Doc): Doc
        Returns:
            RichText: Anchor of mention
        """
        try:
            text = mention_head
            doc_start = tokens_idx[tokens_str.split().index(text)]
            offset = 0
            for sen in doc.sents:
                if tokens_str in str(sen):
                    break
                offset += len(str(sen).split())
            start = doc_start - offset
            end = start + len(mention_head.split()) - 1
            words = text.split()
            return RichText(text, start, end, words)
        except ValueError as e:
            print(f"{e}", tokens_str, tokens_idx, mention_head)
            return

    def parse_mention_extent(
        self,
        doc: Doc,
        tokens_str: str,
    ) -> RichText:
        """Processing mention extent as a RichText instance

        Args:
            doc (Doc):
            tokens_str (str): Index of 'mention span text' within a document

        Returns:
            RichText: Extent of mention
        """

        text = ""
        start, end = 0, 0
        for sen in doc.sents:
            if tokens_str in str(sen):
                text = str(sen)
                break
            start += len(sen)
        words = text.split()
        end = start + len(words) - 1
        return RichText(text, start, end, words)

    def parse_mention_extent_for_special_case(self, context: List[str], tokens_index: List[int]) -> RichText:
        sent_pos_char = [".", "?", "!"]

        text = ""
        start, end = tokens_index[0], tokens_index[-1]
        pre_off, after_off = 0, 0
        for i, t in enumerate(context[:start][::-1]):
            if t not in sent_pos_char:
                pre_off -= 1
            else:
                break

        for i, t in enumerate(context[end:]):
            if t not in sent_pos_char:
                after_off += 1
            else:
                break
        start += pre_off
        end += after_off
        words = context[start : end + 1]
        text = " ".join(words)
        return RichText(text, start, end - 1, words)

    def load_mentions(self):
        model_name = "en_core_web_sm"
        lang_model = spacy.load(model_name, exclude=["ner"])

        raw_data = self._open_file()
        for mention_info in raw_data:
            spaces = [True for _ in mention_info["mention_context"]]
            spaces[-1] = False
            raw_doc = Doc(lang_model.vocab, words=mention_info["mention_context"], spaces=spaces)
            doc = lang_model(raw_doc)

            doc_id = mention_info["doc_id"]

            mention_id = mention_info["mention_id"]
            # Extent
            extent = self.parse_mention_extent(doc=doc, tokens_str=mention_info["tokens_str"])
            if extent.text == "":  # special cases
                extent = self.parse_mention_extent_for_special_case(
                    context=mention_info["mention_context"], tokens_index=mention_info["tokens_number"]
                )

            # Anchor
            anchor = self.parse_mention_anchor(
                mention_info["tokens_str"],
                mention_info["tokens_number"],
                mention_info["mention_head"],
                doc,
            )
            anchor.start = extent.start + anchor.start
            anchor.end = extent.start + anchor.end
            #
            arguments = None
            meta = {
                "mention_head_pos": mention_info["mention_head_pos"],
                "event_id": mention_info["coref_chain"],
            }
            mention = Mention(doc_id, mention_id, extent, anchor, arguments, meta)
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
