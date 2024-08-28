#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
结果：
1. 能复现效果，但是得到的用于训练下游任务的pair比原来论文报告的少很多

nohup sh run.sh 5 easyecr/pipeline/lemma_distance_pipeline.py --config_filename lemma_distance_ecbplus > lemma_distance_ecbplus.log 2>&1 &



Date: 2023/11/28 11:01
"""

from typing import Optional
import argparse
import os

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    print(f'set CUDA_VISIBLE_DEVICES: {os.environ["CUDA_VISIBLE_DEVICES"]}')
else:
    print(f'CUDA_VISIBLE_DEVICES: {os.environ["CUDA_VISIBLE_DEVICES"]}')
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from omegaconf import OmegaConf

from easyecr.common import common_path
from easyecr.ecr_model.framework.ecr_framework import EcrFramework
from easyecr.ecr_model.model.ecr_model import EcrModel
from easyecr.ecr_model.model.pl_ecr_models.lemma_distance_ecr_model import LemmaDistanceEcrModel
from easyecr.ecr_model.model.pl_ecr_models.end_to_end import EndToEndEcrModel
from easyecr.ecr_model.cluster.cluster_model import EcrAgglomerativeClustering
from easyecr.ecr_model.cluster.cluster_model import EcrConnectedComponent
from easyecr.ecr_evaluate.ecr_evaluate import Evaluator
from easyecr.ecr_model.ecr_tagger.ecr_tagger import EcrTagger
from easyecr.utils import object_utils
from easyecr.ecr_model.model.simple_ecr_models.lemma_distance import LemmaDistance
from easyecr.ecr_model.ecr_tagger.doc_tagger import DocTagger
from easyecr.ecr_data.data_converter.data_converter import SplitDataConverter
from easyecr.utils import load_data


def get_dataset(
    config_name: str,
    cache_dir: str,
    dataset_name: str,
    debug: bool = False,
    config=None,
    raw_data_cache_dir: str = "/home/nobody/project/EasyECR/cache",
    train_path: str = "",
    dev_path: str = "",
    test_path: str = "",
    total_path: str = "",
):
    """

    Args:
        config_name:
        use_cache:
        cache_dir:
        total_path:
        config:

    Returns:

    """

    os.makedirs(cache_dir, exist_ok=True)
    train_data_cache_path = os.path.join(cache_dir, f"{config_name}.train.pkl")
    dev_data_cache_path = os.path.join(cache_dir, f"{config_name}.dev.pkl")
    test_data_cache_path = os.path.join(cache_dir, f"{config_name}.test.pkl")
    if (
        os.path.exists(train_data_cache_path)
        and os.path.exists(dev_data_cache_path)
        and os.path.exists(test_data_cache_path)
    ):
        train_data = object_utils.load(train_data_cache_path)
        dev_data = object_utils.load(dev_data_cache_path)
        test_data = object_utils.load(test_data_cache_path)
    else:
        train_data, dev_data, test_data = load_data.get_dataset(
            dataset_name,
            cache_dir=raw_data_cache_dir,
            train_path=train_path,
            dev_path=dev_path,
            test_path=test_path,
            total_path=total_path,
        )
        object_utils.save(train_data, train_data_cache_path)
        object_utils.save(dev_data, dev_data_cache_path)
        object_utils.save(test_data, test_data_cache_path)

    if debug:
        train_data.reduce_mentions_for_debugging()
        dev_data.reduce_mentions_for_debugging()
        test_data.reduce_mentions_for_debugging()
    return train_data, dev_data, test_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_filename", default="lemma_distance_ecbplus", type=str)
    args = parser.parse_args()

    config_filename = args.config_filename
    config_filepath = os.path.join(common_path.project_dir, f"config/model_dataset/{config_filename}.yaml")
    config = OmegaConf.load(config_filepath)

    dataset_name = config["ecr_data"]["dataset_name"]
    train_data, dev_data, test_data = get_dataset(
        config_filename,
        cache_dir=config["common"]["cache_dir"],
        dataset_name=dataset_name,
        debug=False,
        config=config,
        total_path=config["ecr_data"]["total_path"],
    )

    doc_tagger = DocTagger()
    train_data = doc_tagger.predict(train_data, config["DocTagger"]["output_tag"])
    dev_data = doc_tagger.predict(dev_data, config["DocTagger"]["output_tag"])
    test_data = doc_tagger.predict(test_data, config["DocTagger"]["output_tag"])

    lemma_distance_ecr_model = LemmaDistance(**config["EcrFramework1"]["parameters"]["ecr_model"]["parameters"])
    evaluator = Evaluator(**config["EcrFramework1"]["parameters"]["evaluator"]["parameters"])
    simple_cluster_model = EcrConnectedComponent(**config["EcrFramework1"]["parameters"]["cluster_model"]["parameters"])
    framework = EcrFramework(
        config["EcrFramework1"]["parameters"]["predict_topic"],
        config["EcrFramework1"]["parameters"]["evaluate_topic"],
        config["EcrFramework1"]["parameters"]["main_metric"],
        lemma_distance_ecr_model,
        config["EcrFramework1"]["parameters"]["ecr_model_output_tag"],
        simple_cluster_model,
        evaluator,
    )

    mode = config["common"]["mode"]
    if mode == "train":
        framework.train(train_data, dev_data, output_tag=config["EcrFramework1"]["output_tag"])
        metrics = framework.evaluate(test_data, output_tag=config["EcrFramework1"]["output_tag"])
        print(metrics)
    elif mode == "evaluate":
        metrics = framework.evaluate(test_data, output_tag=config["EcrFramework1"]["output_tag"])
        print(metrics)
    elif mode == "predict":
        result = framework.predict(test_data, output_tag=config["EcrFramework1"]["output_tag"])
    else:
        raise NotImplementedError(mode)
    print("end")
