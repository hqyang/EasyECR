from typing import List
from typing import Dict
from typing import Optional
from typing import Any
from typing import Tuple
from typing import Set
import json
from multiprocessing import Process
import os
import time

from tqdm import tqdm
from easyecr.utils import object_utils
from easyecr.utils import file_utils
from easyecr.common import common_path


class RichText:
    def __init__(self, text: str, start: int, end: int, words: Optional[List[str]] = None):
        """

        :param text:
        :param start:
        :param end:
        :param words: 当文档是按照word粒度进行索引时，才会用到
        """
        self.text = text
        self.start = start
        self.end = end
        self.words = words

    def __repr__(self):
        result = f"{self.text} {self.start} {self.end}"
        return result


class Mention:
    # meta里的预定义字段
    basic_topic_tag_name = "basic_topic"
    basic_topic_tag_value = "all"
    mention_repr_tag_name = "repr"
    mention_distance_tag_name = "distance"
    mention_label_tag_name = "event_id"
    mention_predict_tag_name = "event_id_pred"

    def __init__(
        self,
        doc_id: str,
        mention_id: str,
        extent: RichText,
        anchor: RichText,
        arguments: Optional[List[Dict[str, RichText]]] = None,
        meta: Optional[Dict[str, str]] = None,
    ):
        """

        Args:
            doc_id:
            mention_id:
            extent:
            anchor:
            arguments:
            meta:
                预留字段:
                    basic_topic: all
        """
        self.doc_id = doc_id
        self.mention_id = mention_id
        self.extent = extent
        self.anchor = anchor
        self.arguments = arguments
        self.meta = meta
        self._init()

    def _init(self):
        """对mention实例做一些初始化工作

        Returns:

        """
        if self.meta is None:
            self.meta = {}
        self.meta[Mention.basic_topic_tag_name] = Mention.basic_topic_tag_value

    def get_tag(self, name: str):
        """

        Args:
            name:

        Returns:

        """
        result = self.meta[name]
        return result

    def contain(self, tag_name: str) -> bool:
        """

        Args:
            tag_name:

        Returns:

        """
        result = tag_name in self.meta
        return result

    def get_tags_by_prefix(self, prefix: str) -> List[str]:
        """

        Args:
            prefix:

        Returns:

        """
        result = []
        if self.meta is None:
            return result
        for name in self.meta:
            if name.startswith(prefix):
                result.append(name)
        return result

    def add_tag(self, name: str, value: Any):
        """

        Args:
            name:
            value:

        Returns:

        """
        self.meta[name] = value

    def __repr__(self):
        result = f"{self.anchor}    {self.extent}"
        return result


class Event:
    """ """

    def __init__(
        self,
        event_id: str,
        mentions: List[Mention],
        meta: Optional[Dict[str, str]] = None,
        arguments: Optional[List[Dict[str, RichText]]] = None,
    ):
        """

        :param event_id:
        :param mentions:
        :param meta:
        :param arguments:
        """
        self.event_id = event_id
        self.mentions = mentions
        self.meta = meta
        self.arguments = arguments

    def __repr__(self):
        result = self.event_id
        return result


class Document:
    # meta里的预定义字段
    token_tag_name = "doc_token"
    doc_id_tag_name = "doc_id"

    def __init__(self, doc_id: str, text: str, meta: Optional[Dict[str, str]] = None):
        """
        meta may include topic, subtopic, cluster
        :param doc_id:
        :param text:
        :param meta:
        """
        self.doc_id = doc_id
        self.text = text
        self.meta = meta
        self.add_tag(Document.doc_id_tag_name, doc_id)

    def get_tag(self, name: str):
        """

        Args:
            name:

        Returns:

        """
        result = self.meta[name]
        return result

    def contain(self, tag_name: str) -> bool:
        """

        Args:
            tag_name:

        Returns:

        """
        result = tag_name in self.meta
        return result

    def add_tag(self, name: str, value: Any):
        """

        Args:
            name:
            value:

        Returns:

        """
        if self.meta is None:
            self.meta = {}
        self.meta[name] = value

    def get_tags_by_prefix(self, prefix: str) -> List[str]:
        """

        Args:
            prefix:

        Returns:

        """
        result = []
        if self.meta is None:
            return result
        for name in self.meta:
            if name.startswith(prefix):
                result.append(name)
        return result


class EcrData:
    # meta里的预定义字段
    index_type_tag_name = "index_type"
    predict_type_tag_name = "predict_type"

    def __init__(
        self,
        name: str,
        documents: Dict[str, Document],
        mentions: Dict[str, Mention],
        events: Optional[List[Event]] = None,
        meta: Optional[Dict[str, str]] = None,
    ):
        """

        :param name:
        :param documents:
        :param mentions:
        :param events:
        :param meta:
        """
        self.name = name
        self.documents = documents
        self.mentions = mentions
        self.events = events
        self.meta = meta

    def add_event_id(self, mention_tag_name: str = Mention.mention_label_tag_name):
        """根据events字段，给每个mention打上一个所属事件的tag

        :param mention_tag_name:
        :return:
        """
        mention_num = 0
        mention_ids = set()
        for event in self.events:
            for mention in event.mentions:
                mention.add_tag(mention_tag_name, event.event_id)
                mention_num += 1
                mention_ids.add(mention.mention_id)
        assert len(self.mentions) == mention_num

    def add_mention_tag(self, tag_name: str, mention_id_and_value: Dict[str, Any]):
        """

        Args:
            tag_name:
            mention_id_and_value:

        Returns:

        """
        assert len(self.mentions) == len(mention_id_and_value)
        for mention_id, value in mention_id_and_value.items():
            self.mentions[mention_id].meta[tag_name] = value

    def get_mention_tag(self, mention: Mention, tag_name: str):
        """
        mention会继承document的tag
        Args:
            mention:
            tag_name:

        Returns:

        """
        if mention.contain(tag_name):
            result = mention.get_tag(tag_name)
        else:
            result = self.documents[mention.doc_id].get_tag(tag_name)
        return result

    def group_mention_by_tag(self, group_tag_name: str) -> Dict[str, List[Mention]]:
        """

        Args:
            group_tag_name:

        Returns:

        """
        result = {}
        for mention in self.mentions.values():
            tag = self.get_mention_tag(mention, group_tag_name)
            if tag not in result:
                result[tag] = []
            result[tag].append(mention)
        return result

    def get_mention_tags_by_prefix(self, prefix: str) -> List[str]:
        """

        Args:
            prefix:

        Returns:

        """
        result = list(self.mentions.values())[1].get_tags_by_prefix(prefix)
        return result

    def get_mention_context(
        self,
        mention_id: str,
        local_context_type: str = "doc",
        max_local_context_len: int = 200,
        global_context_type: Optional[str] = None,
        first_sentence_num: int = 2,
    ) -> List[Tuple[str, str, str, str]]:
        """

        Args:
            mention_id:
            local_context_type: doc, sentence
            max_local_context_len:
            global_context_type: None, first_sentences
            first_sentence_num:

        Returns: [global_context, pre_context, mention, post_context]

        """
        if self.meta[EcrData.index_type_tag_name] == "word":
            mention = self.mentions[mention_id]
            anchor_start = mention.anchor.start
            anchor_end = mention.anchor.end + 1
            if global_context_type == "first_sentences":
                context_words = [
                    e["text"]
                    for e in self.documents[mention.doc_id].meta["doc_token"]
                    if e["sentence"] in [str(s_id) for s_id in range(first_sentence_num)]
                ]
                global_context = " ".join(context_words)
            else:
                global_context = ""

            if local_context_type == "doc":
                doc = self.documents[mention.doc_id]
                tokens = doc.meta[Document.token_tag_name]
                half_max_context_len = int(max_local_context_len / 2)
                pre_context = " ".join([e["text"] for e in tokens[:anchor_start][half_max_context_len * -1 :]])
                post_context = " ".join([e["text"] for e in tokens[anchor_end:][:half_max_context_len]])
            else:
                extent_start = mention.extent.start
                extent_words = mention.extent.words
                pre_words = extent_words[: anchor_start - extent_start]
                post_words = extent_words[anchor_end - extent_start :]
                pre_context = " ".join(pre_words)
                post_context = " ".join(post_words)
            result = [global_context, pre_context, mention.anchor.text, post_context]
            return result
        elif self.meta[EcrData.index_type_tag_name] == "char":
            raise NotImplementedError(self.meta[EcrData.index_type_tag_name])
        else:
            raise NotImplementedError(self.meta[EcrData.index_type_tag_name])

    @staticmethod
    def from_json(filepath: str) -> "EcrData":
        """

        :param filepath:
        :return:
        """
        pass

    def to_json(self, filepath: str):
        """

        :param filepath:
        :return:
        """
        obj_container = object_utils.convert_to_dict(self)
        file_utils.write_lines([json.dumps(obj_container, ensure_ascii=False)], filepath)

    def get_singleton_mention_ids(self):
        """

        Returns:

        """
        result = set()
        for event in self.events:
            mentions = event.mentions
            if len(mentions) == 1:
                result.add(mentions[0].mention_id)
        return result

    def get_mention_pairs(
        self, within_tag_name: str = Mention.basic_topic_tag_name, include_singleton: bool = True
    ) -> List[Tuple[Mention, Mention]]:
        """

        Args:
            within_tag_name:
            include_singleton:

        Returns:

        """
        result = []
        mentions = list(self.mentions.values())
        singleton_mention_ids = self.get_singleton_mention_ids() if not include_singleton else set()
        for i, mention1 in tqdm(enumerate(mentions)):
            for j, mention2 in enumerate(mentions):
                if (
                    j <= i
                    or self.get_mention_tag(mention1, within_tag_name)
                    != self.get_mention_tag(mention2, within_tag_name)
                    or (
                        not include_singleton
                        and (
                            mention1.mention_id in singleton_mention_ids or mention2.mention_id in singleton_mention_ids
                        )
                    )
                ):
                    continue
                result.append([mention1, mention2])
        return result

    @staticmethod
    def generate_pairs_for_target_mention(
        ecr_data: "EcrData",
        indices: List[int],
        mentions: List[Mention],
        include_singleton: bool,
        within_tag_name: str,
        singleton_mention_ids: Set[str],
        output_filepath: str,
    ):
        """

        Args:
            ecr_data:
            indices:
            mentions:
            include_singleton:
            within_tag_name:
            singleton_mention_ids:
            output_filepath:
        Returns:

        """
        result = []
        for target_mention_index in indices:
            mention1 = mentions[target_mention_index]
            for j, mention2 in enumerate(mentions):
                if (
                    j <= target_mention_index
                    or ecr_data.get_mention_tag(mention1, within_tag_name)
                    != ecr_data.get_mention_tag(mention2, within_tag_name)
                    or (
                        not include_singleton
                        and (
                            mention1.mention_id in singleton_mention_ids or mention2.mention_id in singleton_mention_ids
                        )
                    )
                ):
                    continue
                result.append([mention1, mention2])
        object_utils.save(result, output_filepath)

    @staticmethod
    def generate_pairs_for_target_mention_bridge(parameters):
        """

        Args:
            parameters:

        Returns:

        """
        EcrData.generate_pairs_for_target_mention(*parameters)

    def get_mention_pairs_parallel(
        self, within_tag_name: str = Mention.basic_topic_tag_name, include_singleton: bool = True, thread_num: int = 10
    ) -> List[Tuple[Mention, Mention]]:
        """

        Args:
            within_tag_name:
            include_singleton:
            thread_num:

        Returns:

        """
        result = []
        mentions = list(self.mentions.values())
        singleton_mention_ids = self.get_singleton_mention_ids() if not include_singleton else set()

        mention_groups = [[] for _ in range(thread_num)]
        for i in range(len(mentions)):
            mention_groups[i % thread_num].append(i)

        processes = []
        filepaths = []
        base_dir = os.path.join(common_path.cache_dir, "pair")
        os.makedirs(base_dir, exist_ok=True)
        for i, group in enumerate(mention_groups):
            filepath = os.path.join(base_dir, f"{int(time.time())}_{i}.pkl")
            filepaths.append(filepath)
            process = Process(
                target=EcrData.generate_pairs_for_target_mention,
                args=(self, group, mentions, include_singleton, within_tag_name, singleton_mention_ids, filepath),
            )
            processes.append(process)

        for p in processes:
            p.start()

        for p in processes:
            p.join()

        for filepath in filepaths:
            part_pairs = object_utils.load(filepath)
            result.extend(part_pairs)
            os.remove(filepath)

        return result

    def add_mention_distances(self, id_id_distance: Dict[str, Dict[str, float]], output_tag: str):
        """

        Args:
            id_id_distance: mention之间的距离
            output_tag:

        Returns:

        """
        for mention_id, id_distance in id_id_distance.items():
            self.mentions[mention_id].add_tag(output_tag, id_distance)

    def reduce_mentions_for_debugging(self, keep_mention_num: int = 100):
        """

        Args:
            keep_mention_num:

        Returns:

        """

        mentions_to_remove = []
        if self.events:
            kept_events = []
            kept_mention_num = 0
            for event in self.events:
                if kept_mention_num >= keep_mention_num:
                    mentions_to_remove.extend([e.mention_id for e in event.mentions])
                else:
                    kept_events.append(event)
                    kept_mention_num += len(event.mentions)
            self.events = kept_events
        else:
            for i, mention_id in enumerate(self.mentions.values()):
                if i >= keep_mention_num:
                    mentions_to_remove.append(mention_id)

        for mention_id in mentions_to_remove:
            self.mentions.pop(mention_id)
