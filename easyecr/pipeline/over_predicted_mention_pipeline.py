#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""

nohup sh run.sh 3 easyecr/pipeline/over_predicted_mention_pipeline.py --config_filename end_to_end_ecbplus > end_to_end_ecbplus.log 2>&1 &

nohup sh run.sh 3 easyecr/pipeline/over_predicted_mention_pipeline.py --config_filename end_to_end_wec > end_to_end_wec.log 2>&1 &




Date: 2023/11/28 11:01
"""

from typing import Optional
import argparse
import os

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print(f'set CUDA_VISIBLE_DEVICES: {os.environ["CUDA_VISIBLE_DEVICES"]}')
else:
    print(f'CUDA_VISIBLE_DEVICES: {os.environ["CUDA_VISIBLE_DEVICES"]}')
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# os.environ["no_proxy"] = 'localhost,127.0.0.1'
# os.environ["http_proxy"] = 'http://192.168.1.174:12798'
# os.environ["https_proxy"] = 'http://192.168.1.174:12798'
# os.environ["REQUESTS_CA_BUNDLE"] = '/etc/ssl/certs/ca-certificates.crt'
# os.environ["SSL_CERT_FILE"] = '/etc/ssl/certs/ca-certificates.crt'

from omegaconf import OmegaConf

from easyecr.common import common_path
from easyecr.ecr_model.framework.ecr_framework import EcrFramework
from easyecr.ecr_model.model.ecr_model import EcrModel
from easyecr.ecr_model.model.pl_ecr_models.crl_ecr_model import ContrastiveRepresentationLearningEcrModel
from easyecr.ecr_model.model.pl_ecr_models.end_to_end import EndToEndEcrModel
from easyecr.ecr_model.cluster.cluster_model import EcrAgglomerativeClustering
from easyecr.ecr_data.data_converter.data_converter import SplitDataConverter
from easyecr.ecr_evaluate.ecr_evaluate import Evaluator
from easyecr.ecr_model.ecr_tagger.ecr_tagger import EcrTagger
from easyecr.ecr_model.ecr_tagger.contextualized_representation_tagger import ContextualizedRepresentationTagger
from easyecr.utils import object_utils
from easyecr.ecr_data.datasets.wec.wec_eng import WECEngDataset
from easyecr.utils import load_data
from easyecr.ecr_model.sample_generator.sample_generator import SimpleGenerator


def get_dataset(
    cache_dir: str,
    dataset_name: str,
    debug: bool = False,
    raw_data_cache_dir: str = "/home/nobody/code/ecr-code/end_to_end_wec_cache",
    train_path: str = "",
    dev_path: str = "",
    test_path: str = "",
    total_path: str = "",
    tagger=None,
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
    train_data_cache_path = os.path.join(cache_dir, f"{dataset_name}.train.pkl")
    dev_data_cache_path = os.path.join(cache_dir, f"{dataset_name}.dev.pkl")
    test_data_cache_path = os.path.join(cache_dir, f"{dataset_name}.test.pkl")
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

        if tagger:
            train_data = tagger.predict(train_data, output_tag="mention")
            dev_data = tagger.predict(dev_data, output_tag="mention")
            test_data = tagger.predict(test_data, output_tag="mention")

        object_utils.save(train_data, train_data_cache_path)
        object_utils.save(dev_data, dev_data_cache_path)
        object_utils.save(test_data, test_data_cache_path)

    if debug:
        keep_mention_num = 70
        train_data.reduce_mentions_for_debugging(keep_mention_num=keep_mention_num)
        dev_data.reduce_mentions_for_debugging(keep_mention_num=keep_mention_num)
        test_data.reduce_mentions_for_debugging(keep_mention_num=keep_mention_num)
    return train_data, dev_data, test_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_filename",
        default="end_to_end_fcct",
        choices=["end_to_end_ecbplus", "end_to_end_gvc", "end_to_end_wec", "end_to_end_fcct", "end_to_end_fcc"],
        type=str,
    )
    args = parser.parse_args()

    config_filename = args.config_filename
    config_filepath = os.path.join(common_path.project_dir, f"config/model_dataset/{config_filename}.yaml")
    config = OmegaConf.load(config_filepath)
    config = OmegaConf.to_container(config, resolve=True)

    mode = config["common"]["mode"]

    dataset_name = config["ecr_data"]["dataset_name"]
    ecr_tagger: EcrTagger = ContextualizedRepresentationTagger(
        config["tagger"]["max_surrounding_contx"], config["tagger"]["transformer_model"]
    )
    train_data, dev_data, test_data = get_dataset(
        cache_dir=config["common"]["cache_dir"],
        dataset_name=dataset_name,
        debug=False,
        raw_data_cache_dir=config["common"]["cache_dir"],
        train_path=config["ecr_data"]["train_path"],
        dev_path=config["ecr_data"]["dev_path"],
        test_path=config["ecr_data"]["test_path"],
        total_path=config["ecr_data"]["total_path"],
        tagger=ecr_tagger,
    )

    ecr_model = EndToEndEcrModel(
        mode=mode,
        model_dir=config["EcrFramework1"]["parameters"]["ecr_model"]["parameters"]["common"]["model_dir"],
        model_filename=config_filename,
        trainer_parameters=config["EcrFramework1"]["parameters"]["ecr_model"]["parameters"]["trainer_parameters"],
        conf=config["EcrFramework1"]["parameters"]["ecr_model"]["parameters"],
    )
    evaluator = Evaluator(**config["EcrFramework1"]["parameters"]["evaluator"]["parameters"])
    cluster_model = EcrAgglomerativeClustering(**config["EcrFramework1"]["parameters"]["cluster_model"]["parameters"])
    framework = EcrFramework(
        config["EcrFramework1"]["parameters"]["predict_topic"],
        config["EcrFramework1"]["parameters"]["evaluate_topic"],
        config["EcrFramework1"]["parameters"]["main_metric"],
        ecr_model,
        config["EcrFramework1"]["parameters"]["ecr_model_output_tag"],
        cluster_model,
        evaluator,
    )
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


# python over_predicted_mention_pipeline.py --config_filename end_to_end_gvc
# python over_predicted_mention_pipeline.py --config_filename end_to_end_ecbplus
# python over_predicted_mention_pipeline.py --config_filename end_to_end_fcct
# python over_predicted_mention_pipeline.py --config_filename end_to_end_wec
