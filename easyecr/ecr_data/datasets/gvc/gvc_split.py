import os
import re
import pandas as pd


class GVCDatasetSplit:
    def __init__(self, from_directory: str, split_info_directory: str) -> None:
        self.split_info_directory = split_info_directory
        self.from_directory = from_directory
        self.doc2event_path = os.path.join(split_info_directory, "gvc_doc_to_event.csv")

    def split_helper(self):
        train_df = None
        dev_df = None
        test_df = None

        doc2event_df = pd.read_csv(self.doc2event_path)
        train_df = pd.read_csv(os.path.join(self.split_info_directory, f"train.csv"))
        train_merge_df = pd.merge(doc2event_df, train_df, on="event-id")
        dev_df = pd.read_csv(os.path.join(self.split_info_directory, f"dev.csv"))
        dev_merge_df = pd.merge(doc2event_df, dev_df, on="event-id")
        test_df = pd.read_csv(os.path.join(self.split_info_directory, f"test.csv"))
        test_merge_df = pd.merge(doc2event_df, test_df, on="event-id")
        with open(os.path.join(self.from_directory, "gold.conll"), "r") as input_file, open(
            os.path.join(self.from_directory, "train.conll"), "+a"
        ) as train_file, open(os.path.join(self.from_directory, "dev.conll"), "+a") as dev_file, open(
            os.path.join(self.from_directory, "test.conll"), "+a"
        ) as test_file:
            flag = ""
            doc = []
            for line in input_file.readlines():
                doc.append(line)
                if "#begin document" in line:
                    doc_id = re.search(r"\((.*?)\)", line).group(1)
                    if doc_id in train_merge_df["doc-id"].values:
                        flag = "train"
                    elif doc_id in dev_merge_df["doc-id"].values:
                        flag = "dev"
                    elif doc_id in test_merge_df["doc-id"].values:
                        flag = "test"
                    else:
                        pass
                elif "#end document" in line:
                    if flag == "train":
                        for l in doc:
                            train_file.write(l)
                    elif flag == "dev":
                        for l in doc:
                            dev_file.write(l)
                    elif flag == "test":
                        for l in doc:
                            test_file.write(l)
                    else:
                        pass
                    doc = []
                else:
                    pass
