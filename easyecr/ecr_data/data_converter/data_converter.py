#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""



Date: 2023/10/18 14:57
"""
from typing import List
from typing import Dict
from collections import defaultdict

from sklearn.model_selection import train_test_split

from easyecr.ecr_data.data_structure.data_structure import EcrData
from easyecr.ecr_data.data_structure.data_structure import Event
from easyecr.ecr_data.datasets.ace.ace import ACE2005Dataset
from easyecr.ecr_data.datasets.ecb_plus.ecb_plus import ECBPlusDataset
from easyecr.ecr_data.datasets.idea_car.idea_car import IDEACarDataset
from easyecr.ecr_data.datasets.fcc.fcc import FCCDataset
from easyecr.ecr_data.datasets.fcc_t.fcc_t import FCCTDataset
from easyecr.ecr_data.datasets.gvc.gvc import GVCDataset
from easyecr.ecr_data.datasets.kbp.kbp_2015 import KBPEng2015Dataset
from easyecr.ecr_data.datasets.kbp.kbp_2016 import KBP2016Dataset
from easyecr.ecr_data.datasets.kbp.kbp_2017 import KBPEng2017Dataset
from easyecr.ecr_data.datasets.kbp.kbp_eng_mix import KBPEngMixDataset
from easyecr.ecr_data.datasets.maven.maven import MAVENDataset
from easyecr.ecr_data.datasets.maven_ere.maven_ere import MAVENEREDataset
from easyecr.ecr_data.datasets.wec.wec_eng import WECEngDataset


class DataConverter:
    """ """

    @staticmethod
    def from_directory(dataset_name: str, directory: str) -> "EcrData":
        """

        :param dataset_name:
        :param directory:
        :return:
        """
        if dataset_name in ["ace2005eng"]:
            dataset = ACE2005Dataset(dataset_name, directory)
        elif dataset_name in ["ecbplus"]:
            dataset = ECBPlusDataset(dataset_name, directory)
        elif dataset_name in ["ideacar"]:
            dataset = IDEACarDataset(dataset_name, directory)
        elif dataset_name in ["fcc"]:
            dataset = FCCDataset(dataset_name, directory)
        elif dataset_name in ["fcct"]:
            dataset = FCCTDataset(dataset_name, directory)
        elif dataset_name in ["gvc"]:
            dataset = GVCDataset(dataset_name, directory)
        elif dataset_name in ["kbp2015eng"]:
            dataset = KBPEng2015Dataset(dataset_name, directory)
        elif dataset_name in ["kbp2016cmn", "kbp2016eng", "kbp2016spa"]:
            dataset = KBP2016Dataset(dataset_name, directory)
        elif dataset_name in ["kbp2017eng"]:
            dataset = KBPEng2017Dataset(dataset_name, directory)
        elif dataset_name in ["kbpmix"]:
            dataset = KBPEngMixDataset(dataset_name, directory)
        elif dataset_name in ["maven"]:
            dataset = MAVENDataset(dataset_name, directory)
        elif dataset_name in ["mavenere"]:
            dataset = MAVENEREDataset(dataset_name, directory)
        elif dataset_name in ["wec"]:
            dataset = WECEngDataset(dataset_name, directory)
        else:
            raise NotImplementedError(dataset_name)
        result = dataset.to_ecr_data()
        return result


class SplitDataConverter:
    @staticmethod
    def split(dataset_name: str, train_path: str = "", dev_path: str = "", test_path: str = "", total_path: str = ""):
        """If the data set provider has provided the divided dataset, specify train_path, dev_path,
        and test_path in sequence; if the data set does not provide this, you only need to specify
        total_path and empirical division will be used.

        Args:
            dataset_name (str): dataset_name
            train_path (str, optional): The path of raw train dataset. Defaults to "".
            dev_path (str, optional): The path of raw dev dataset. Defaults to "".
            test_path (str, optional): The path of raw test dataset. Defaults to "".
            total_path (str, optional): The path of raw total dataset. Defaults to "".

        Returns:
            _type_: _description_
        """
        if total_path:
            if dataset_name in ["ace2005eng", "kbp2016cmn", "kbp2016eng", "kbp2016spa", "kbp2017eng"]:
                split_dataset = SplitAdapter.split_with_prop(dataset_name, total_path)
            elif dataset_name in ["ecbplus"]:
                split_dataset = SplitAdapter.split_with_topic(dataset_name, total_path)
            else:
                raise NotImplementedError(dataset_name)
        elif train_path or dev_path or test_path:
            selected_paths = {
                key: path
                for key, path in {"train": train_path, "dev": dev_path, "test": test_path}.items()
                if path != ""
            }
            split_dataset = SplitAdapter.split_with_golden(dataset_name, **selected_paths)
        else:
            raise NotImplementedError
        return split_dataset


class SplitAdapter:
    @staticmethod
    def split_with_golden(
        dataset_name: str,
        **kwargs,
    ):
        result = []
        for key, value in kwargs.items():
            result.append(DataConverter.from_directory(dataset_name=dataset_name, directory=value))
        return result

    @staticmethod
    def split_with_topic(
        dataset_name: str,
        dir_or_path: str,
        topic_list: Dict[str, list] = {
            "train": [
                "1",
                "10",
                "11",
                "13",
                "14",
                "16",
                "19",
                "20",
                "22",
                "24",
                "25",
                "26",
                "27",
                "28",
                "29",
                "3",
                "30",
                "31",
                "32",
                "33",
                "4",
                "6",
                "7",
                "8",
                "9",
            ],
            "dev": ["12", "18", "2", "21", "23", "34", "35", "5"],
            "test": ["36", "37", "38", "39", "40", "41", "42", "43", "44", "45"],
        },
    ):
        dataset = DataConverter.from_directory(dataset_name=dataset_name, directory=dir_or_path)
        documents = dataset.documents
        mentions = dataset.mentions
        events = dataset.events
        meta = dataset.meta

        train_documents, dev_documents, test_documents = {}, {}, {}
        train_mentions, dev_mentions, test_mentions = {}, {}, {}

        for doc_id, doc in documents.items():
            if doc.meta["doc_topic"] in topic_list["train"]:
                train_documents[doc_id] = doc
            elif doc.meta["doc_topic"] in topic_list["dev"]:
                dev_documents[doc_id] = doc
            else:
                test_documents[doc_id] = doc

        for m_id, mention in mentions.items():
            if mention.doc_id in train_documents.keys():
                train_mentions[m_id] = mention
            elif mention.doc_id in dev_documents.keys():
                dev_mentions[m_id] = mention
            else:
                test_mentions[m_id] = mention

        def merge_mentions(mentions: dict) -> List[Event]:
            event2mentions = defaultdict(list)
            split_events = []
            for _, mention in mentions.items():
                event_id = mention.meta["event_id"]
                event2mentions[event_id].append(mention)
            for event_id, mentions in event2mentions.items():
                meta = None
                for event in events:
                    if event.event_id == event_id:
                        meta = event.meta
                        break
                event = Event(event_id, mentions, meta)
                split_events.append(event)
            return split_events

        train_events = merge_mentions(train_mentions)
        dev_events = merge_mentions(dev_mentions)
        test_events = merge_mentions(test_mentions)
        train_ecrdata = EcrData(
            name=dataset_name, documents=train_documents, mentions=train_mentions, events=train_events, meta=meta
        )
        dev_ecrdata = EcrData(
            name=dataset_name, documents=dev_documents, mentions=dev_mentions, events=dev_events, meta=meta
        )
        test_ecrdata = EcrData(
            name=dataset_name, documents=test_documents, mentions=test_mentions, events=test_events, meta=meta
        )
        return [train_ecrdata, dev_ecrdata, test_ecrdata]

    @staticmethod
    def split_with_prop(dataset_name: str, dir_or_path: str, proportion: tuple = (0.8, 0.1, 0.1)):
        dataset = DataConverter.from_directory(dataset_name=dataset_name, directory=dir_or_path)
        documents = dataset.documents
        doc_ids = list(documents.keys())
        mentions = dataset.mentions
        events = dataset.events
        meta = dataset.meta

        num_doc = len(documents)
        train_doc_index, test_doc_index = train_test_split(
            range(num_doc),
            test_size=proportion[1] + proportion[2],
            train_size=proportion[0],
            shuffle=False,
        )
        dev_doc_index, test_doc_index = train_test_split(
            test_doc_index,
            test_size=proportion[1] / (proportion[1] + proportion[2]),
            shuffle=False,
        )
        train_documents, dev_documents, test_documents = {}, {}, {}
        train_mentions, dev_mentions, test_mentions = {}, {}, {}

        train_doc_ids = [doc_ids[i] for i in train_doc_index]
        dev_doc_ids = [doc_ids[i] for i in dev_doc_index]
        test_doc_ids = [doc_ids[i] for i in test_doc_index]
        for doc_id in train_doc_ids:
            train_documents[doc_id] = documents[doc_id]
        for doc_id in dev_doc_ids:
            dev_documents[doc_id] = documents[doc_id]
        for doc_id in test_doc_ids:
            test_documents[doc_id] = documents[doc_id]

        for m_id, mention in mentions.items():
            doc_id = mention.doc_id
            if doc_id in train_doc_ids:
                train_mentions[m_id] = mention
            elif doc_id in dev_doc_ids:
                dev_mentions[m_id] = mention
            elif doc_id in test_doc_ids:
                test_mentions[m_id] = mention
            else:
                pass

        def merge_mentions(mentions: dict) -> List[Event]:
            event2mentions = defaultdict(list)
            split_events = []
            for _, mention in mentions.items():
                event_id = mention.meta["event_id"]
                event2mentions[event_id].append(mention)
            for event_id, mentions in event2mentions.items():
                meta = None
                for event in events:
                    if event.event_id == event_id:
                        meta = event.meta
                        break
                event = Event(event_id, mentions, meta)
                split_events.append(event)
            return split_events

        train_events = merge_mentions(train_mentions)
        dev_events = merge_mentions(dev_mentions)
        test_events = merge_mentions(test_mentions)
        train_ecrdata = EcrData(
            name=dataset_name, documents=train_documents, mentions=train_mentions, events=train_events, meta=meta
        )
        dev_ecrdata = EcrData(
            name=dataset_name, documents=dev_documents, mentions=dev_mentions, events=dev_events, meta=meta
        )
        test_ecrdata = EcrData(
            name=dataset_name, documents=test_documents, mentions=test_mentions, events=test_events, meta=meta
        )
        return [train_ecrdata, dev_ecrdata, test_ecrdata]


if __name__ == "__main__":
    # # ACE2005
    # acedataset = SplitDataConverter.split(
    #     dataset_name="ace2005eng", total_path="/data/dev/ecr-data/ace2005/ace_2005_td_v7/data/English/"
    # )
    # # ECBPlus
    # ecbplusdataset = SplitDataConverter.split(
    #     dataset_name="ecbplus", total_path="/data/dev/ecr-data/ECB+_LREC2014"
    # )
    # # FCC
    # fccdataset = SplitDataConverter.split(
    #     dataset_name="fcc",
    #     train_path="/data/dev/ecr-data/fcc/2020-03-18_FCC/train",
    #     dev_path="/data/dev/ecr-data/fcc/2020-03-18_FCC/dev",
    #     test_path="/data/dev/ecr-data/fcc/2020-03-18_FCC/test",
    # )
    # # FCCT
    # fcctdataset = SplitDataConverter.split(
    #     dataset_name="fcct",
    #     train_path="/data/dev/ecr-data/fcc/2020-10-05_FCC-T/train",
    #     dev_path="/data/dev/ecr-data/fcc/2020-10-05_FCC-T/dev",
    #     test_path="/data/dev/ecr-data/fcc/2020-10-05_FCC-T/test",
    # )
    # GVC
    # gvcdataset = SplitDataConverter.split(
    #     dataset_name="gvc",
    #     train_path="/data/dev/ecr-data/GunViolenceCorpus-master/train.conll",
    #     dev_path="/data/dev/ecr-data/GunViolenceCorpus-master/dev.conll",
    #     test_path="/data/dev/ecr-data/GunViolenceCorpus-master/test.conll",
    # )
    # print()
    # # KBP2015
    # kbp2015dataset = SplitDataConverter.split(
    #     dataset_name="kbp2015eng",
    #     train_path="/data/dev/ecr-data/LDC2017E02_TAC_KBP_Event_Nugget_Detection_and_Coreference_Comprehensive_Training_and_Evaluation_Data_2014-2016/data/2015/training",
    #     dev_path="/data/dev/ecr-data/LDC2017E02_TAC_KBP_Event_Nugget_Detection_and_Coreference_Comprehensive_Training_and_Evaluation_Data_2014-2016/data/2015/eval",
    # )
    # # KBP2016CMN
    # kbp2016cmndataset = SplitDataConverter.split(
    #     dataset_name="kbp2016cmn",
    #     total_path="/data/dev/ecr-data/LDC2017E02_TAC_KBP_Event_Nugget_Detection_and_Coreference_Comprehensive_Training_and_Evaluation_Data_2014-2016/data/2016/eval/cmn",
    # )
    # # KBP2016ENG
    # kbp2016engdataset = SplitDataConverter.split(
    #     dataset_name="kbp2016eng",
    #     total_path="/data/dev/ecr-data/LDC2017E02_TAC_KBP_Event_Nugget_Detection_and_Coreference_Comprehensive_Training_and_Evaluation_Data_2014-2016/data/2016/eval/eng",
    # )
    # # KBP2016SPA
    # kbp2016engdataset = SplitDataConverter.split(
    #     dataset_name="kbp2016spa",
    #     total_path="/data/dev/ecr-data/LDC2017E02_TAC_KBP_Event_Nugget_Detection_and_Coreference_Comprehensive_Training_and_Evaluation_Data_2014-2016/data/2016/eval/spa",
    # )
    # # KBP2017ENG
    # kbp2017engdataset = SplitDataConverter.split(
    #     dataset_name="kbp2017eng", total_path="/data/dev/ecr-data/KBP_2017"
    # )
    # # KBPEng151617
    # kbpmixdataset = SplitDataConverter.split(
    #     dataset_name="kbpmix",
    #     train_path="/data/dev/ecr-data/KBP_Eng_201520162017/train_filtered.json",
    #     dev_path="/data/dev/ecr-data/KBP_Eng_201520162017/dev_filtered.json",
    #     test_path="/data/dev/ecr-data/KBP_Eng_201520162017/test_filtered.json",
    # )
    # # MAVEN
    # mavendataset = SplitDataConverter.split(
    #     dataset_name="maven",
    #     train_path="/data/dev/ecr-data/MAVEN/train.jsonl",
    #     dev_path="/data/dev/ecr-data/MAVEN/valid.jsonl",
    #     test_path="/data/dev/ecr-data/MAVEN/test.jsonl",
    # )
    # # MAVENERE
    # maveneredataset = SplitDataConverter.split(
    #     dataset_name="mavenere",
    #     train_path="/data/dev/ecr-data/MAVEN-ERE/train.jsonl",
    #     dev_path="/data/dev/ecr-data/MAVEN-ERE/valid.jsonl",
    #     test_path="/data/dev/ecr-data/MAVEN-ERE/test.jsonl",
    # )

    # WEC
    wecdataset = SplitDataConverter.split(
        dataset_name="wec",
        train_path="/data/dev/ecr-data/WEC-Eng/Train_Event_gold_mentions.json",
        dev_path="/data/dev/ecr-data/WEC-Eng/Dev_Event_gold_mentions_validated.json",
        test_path="/data/dev/ecr-data/WEC-Eng/Test_Event_gold_mentions_validated.json",
    )
