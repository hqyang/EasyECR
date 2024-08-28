import os
import re
from typing import List
from collections import defaultdict

import spacy


class Sentence:
    def __init__(self, filename, text, start):
        self.filename = filename
        self.text = text
        self.start = start


LANG2MODEL = {"eng": "en_core_web_sm", "cmn": "zh_core_web_sm", "spa": "es_core_news_sm"}


class SourceReader:
    def __init__(self, dataset_name: str, folder: str):
        self.folder_path = folder
        if dataset_name in ["kbp2015eng", "kbp2017eng"]:
            self.lang = "eng"
        elif "cmn" in folder:
            self.lang = "cmn"
        elif "eng" in folder:
            self.lang = "eng"
        elif "spa" in folder:
            self.lang = "spa"
        self.nlp = spacy.load(LANG2MODEL[self.lang])

    def read_source_folder_2015(self):
        news_start = ["AFP", "APW", "CNA", "NYT", "WPB", "XIN"]
        folder = os.path.abspath(self.folder_path)
        results = []
        for file in os.listdir(folder):
            if file[0:3] in news_start:  # News
                results.extend(self.read_source_file(file_path=os.path.join(folder, file), is_df=False))
            else:  # Forum
                results.extend(self.read_source_file(file_path=os.path.join(folder, file), is_df=True))
        return self.docid2sents(sentences_info=results)

    def read_source_folded_2016(self, is_df: bool):
        folder = os.path.abspath(self.folder_path)
        results = []
        for file in os.listdir(folder):
            results.extend(self.read_source_file(file_path=os.path.join(folder, file), is_df=is_df))
        return self.docid2sents(sentences_info=results)

    def read_source_file(self, file_path: str, is_df: bool) -> List[Sentence]:
        if is_df:
            return self.forum_article_reader(file_path)
        else:
            return self.news_article_reader(file_path)

    def docid2sents(self, sentences_info: List[Sentence]):
        id2sents = defaultdict(list)
        for sent_info in sentences_info:
            id2sents[sent_info.filename].append({"start": sent_info.start, "text": sent_info.text})

        for sents in id2sents.values():
            sents.sort(key=lambda x: x["start"])
        return id2sents

    def forum_article_reader(self, file_path: str) -> List[Sentence]:
        filename = os.path.basename(file_path)
        filters = [".txt", ".xml", ".mpdf", ".cmp"]
        for w in filters:
            filename = filename.replace(w, "")
        with open(file_path, "r", encoding="utf-8") as file:
            start = 0
            results = []
            flag = ""
            text = ""
            for line in file.readlines():
                line = line.rstrip("\n")
                length = len(line) + 1
                if line.lower().startswith("<post"):
                    flag = "post"
                    start += length
                    continue
                elif line.lower().startswith("<quote") or line.lower().startswith("<headline"):
                    flag = "quote"
                    start += length
                    continue
                elif line.strip().startswith("</quote>") or line.lower().startswith("</headline"):
                    flag = "post"
                    sentences = self.split_sentences(filename, text, start)
                    results.extend(sentences)
                    start += len(text) + length
                    text = ""
                    continue

                if flag == "quote":
                    text += line + " "
                    continue
                elif flag == "post":
                    sentences = self.split_sentences(filename, line, start)
                    results.extend(sentences)
                    start += length
                    continue
                start += length
        return results

    def news_article_reader(self, file_path: str) -> List[Sentence]:
        filename = os.path.basename(file_path)
        filters = [".txt", ".xml", ".mpdf", ".cmp"]
        for w in filters:
            filename = filename.replace(w, "")
        with open(file_path, "r", encoding="utf-8") as file:
            start = 0
            results = []
            flag = ""
            text = ""
            for line in file.readlines():
                line = line.rstrip("\n")
                length = len(line) + 1
                if line.strip().lower() == "<text>":
                    flag = "text"
                    start += length
                    continue
                elif (
                    line.strip().lower() == "<headline>"
                    or line.strip().lower() == "<p>"
                    or line.strip().lower() == "<keyword>"
                ):
                    flag = "para"
                    start += length
                    continue
                elif (
                    line.strip().lower() == "</headline>"
                    or line.strip().lower() == "</p>"
                    or line.strip().lower() == "</keyword>"
                ):
                    flag = ""
                    sentences = self.split_sentences(filename, text, start)
                    results.extend(sentences)
                    start += len(text) + length
                    text = ""
                    continue
                elif line.strip().lower() == "</text>":
                    flag = ""
                    start += length
                    text = ""
                    continue

                if flag == "para":
                    text += line + " "
                    continue
                elif flag == "text":
                    sentences = self.split_sentences(filename, line, start)
                    results.extend(sentences)
                    start += length
                    continue
                start += length
            return results

    def split_sentences(self, filename, text, start) -> List[Sentence]:
        if "<" in text:  # html file
            p_html = re.compile("<[^>]+>", re.IGNORECASE)
            m_html = p_html.sub(lambda m: " " * len(m.group(0)), text)
            text = m_html
            count = 0
            if text.startswith(" "):
                for char in text:
                    if char != " ":
                        break
                    count += 1
            text = text.strip()
            start += count
        # split sentence
        doc = self.nlp(text)
        results = []
        for sent in doc.sents:
            sent_offset = sent.start_char
            sent_text = sent.text.replace("\t", " ")
            if self.lang == ["eng", "spa"]:
                if (
                    self.is_contain_chinese(sent_text)
                    or sent_text.startswith("http")
                    or sent_text.startswith("www.")
                    or self.filter_sentence(sent_text)
                    or not sent_text
                    or len(sent_text) < 3
                ):
                    continue
            elif self.lang == "cmn":
                if (
                    sent_text.startswith("http")
                    or sent_text.startswith("www.")
                    or self.filter_sentence(sent_text)
                    or not sent_text
                    or len(sent_text) < 3
                ):
                    continue
            results.append(Sentence(filename, sent_text, start + sent_offset))
        return results

    def filter_sentence(self, text: str) -> bool:
        stopwords = [
            "P.S.",
            "PS",
            "snip",
            "&amp;",
            "&lt;",
            "&gt;",
            "&nbsp;",
            "&quot;",
            "#",
            "*",
            ".",
            "/",
            "year",
            "day",
            "month",
            "Â",
            "-",
            "[",
            "]",
            "!",
            "?",
            ",",
            ";",
            "(",
            ")",
            ":",
            "~",
            "_",
            "cof",
            "sigh",
            "shrug",
            "and",
            "or",
            "done",
            "URL",
        ]
        for w in stopwords:
            text = text.replace(w, " ")

        text = re.sub(r"\d", " ", text)
        return text.strip() == "" or len(text.strip()) == 1

    def is_contain_chinese(self, text: str) -> bool:
        return bool(re.search(r"[\u4E00-\u9FA5]", text))
