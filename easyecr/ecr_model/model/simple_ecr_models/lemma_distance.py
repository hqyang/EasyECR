#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""



Date: 2023/12/14 17:41
"""
from typing import List
from typing import Tuple
from typing import Set

from omegaconf import DictConfig
import spacy
from spacy.tokens import Doc

from easyecr.ecr_model.model.simple_ecr_models.simple_ecr_models import SimpleDistanceEcrModel
from easyecr.ecr_data.data_structure.data_structure import EcrData
from easyecr.ecr_data.data_structure.data_structure import Mention
from easyecr.ecr_model.sample_generator.sample_generator import SimpleGenerator
from easyecr.utils import log_utils
from easyecr.pipeline import load_lemma_data

logger = log_utils.get_logger(__file__)


class LemmaDistance(SimpleDistanceEcrModel):
    def __init__(
        self, train_topic: str, predict_topic: str, sent_sim_threshold: float, spacy_model_name: str = "en_core_web_sm"
    ):
        """

        Args:
            train_topic:
            predict_topic:
            sent_sim_threshold:
            spacy_model_name:
        """
        super().__init__(train_topic, predict_topic)
        self.sent_sim_threshold = sent_sim_threshold
        self.positive_lemma_pairs = set()
        self.nlp = spacy.load(spacy_model_name, exclude=["ner", "parser"])
        self.lemma_cache = {}
        self.words_cache = {}

        # self.mention_pair_similarity = load_lemma_data.load_mention_pair_and_similarity()

    def word_tokenize(self, text: str) -> List[str]:
        """

        Args:
            text:

        Returns:

        """
        if not text.strip():
            return []

        if text in self.words_cache:
            return self.words_cache[text]

        doc = self.nlp(text)
        result = [e.text.strip() for e in doc if e.text.strip()]
        self.words_cache[text] = result
        return result

    def lemmatize(self, word):
        """

        :param word:
        :return:
        """
        if not word:
            return ""
        if word in self.lemma_cache:
            return self.lemma_cache[word]

        spaces = [False]
        raw_doc = Doc(self.nlp.vocab, words=[word], spaces=spaces)
        doc = self.nlp(raw_doc)

        result = [e.lemma_ for e in doc][0]
        self.lemma_cache[word] = result
        return result

    def get_lemma_pairs(self, samples: List[Tuple[Mention, Mention]]) -> Set[Tuple[str, str]]:
        """

        Args:
            samples:

        Returns:

        """
        result = set()
        for mention1, mention2 in samples:
            text1 = mention1.anchor.text
            text2 = mention2.anchor.text

            lemma1 = self.lemmatize(text1)
            lemma2 = self.lemmatize(text2)
            if lemma1 > lemma2:
                result.add((lemma2, lemma1))
            else:
                result.add((lemma1, lemma2))
        return result

    def train(self, train_data: EcrData, dev_data: EcrData):
        """

        Args:
            train_data:
            dev_data:

        Returns:

        """
        sample_generator = SimpleGenerator(times=0)
        train_samples = sample_generator.generate(train_data, topic_name=self.train_topic)
        logger.info(f"'train samples num: {len(train_samples['positive'] + train_samples['negative'])}")
        # dev_samples = sample_generator.generate(dev_data, topic_name=self.within_tag_name)
        positive_samples = train_samples["positive"]
        lemma_pairs = self.get_lemma_pairs(positive_samples)
        self.positive_lemma_pairs.union(lemma_pairs)

    def compute_jaccard_similarity(self, arr1, arr2):
        """

        Args:
            arr1:
            arr2:

        Returns:

        """
        result = len(set.intersection(arr1, arr2)) / len(set.union(arr1, arr2))
        return result

    def predict_mention_distance(self, mention1: Mention, mention2: Mention, data: EcrData) -> float:
        """

        Args:
            mention1:
            mention2:
            data:

        Returns:

        """
        # key1 = load_lemma_data.get_mention_id(mention1, data)
        # key2 = load_lemma_data.get_mention_id(mention2, data)
        # if (key1, key2) in self.mention_pair_similarity:
        #     similarity = self.mention_pair_similarity[(key1, key2)]
        # elif (key2, key1) in self.mention_pair_similarity:
        #     similarity = self.mention_pair_similarity[(key2, key1)]
        # else:
        #     similarity = 0.0
        # distance = 1 - similarity
        # return distance

        text1 = mention1.anchor.text
        text2 = mention2.anchor.text

        lemma1 = self.lemmatize(text1)
        lemma2 = self.lemmatize(text2)
        lemma_sim = float(lemma1.lower() in text2 or lemma2.lower() in text1 or lemma1.lower() in lemma2.lower())

        if lemma1 > lemma2:
            pair_tuple = (lemma2, lemma1)
        else:
            pair_tuple = (lemma1, lemma2)
        relation_sim = pair_tuple in self.positive_lemma_pairs

        sentence1 = self.word_tokenize(mention1.extent.text)
        sentence2 = self.word_tokenize(mention2.extent.text)
        try:
            sent_sim = self.compute_jaccard_similarity(set(sentence1), set(sentence2))
        except ZeroDivisionError:
            print(sentence1, sentence2)
            print(mention1.mention_id, mention2.mention_id)
            print(mention1.extent.text, mention2.extent.text)

        similarity = (lemma_sim or relation_sim) and sent_sim > self.sent_sim_threshold
        distance = 1 - similarity
        return distance
