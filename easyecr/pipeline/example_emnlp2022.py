import os
import argparse
from typing import Optional

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    print(f'set CUDA_VISIBLE_DEVICES: {os.environ["CUDA_VISIBLE_DEVICES"]}')
else:
    print(f'CUDA_VISIBLE_DEVICES: {os.environ["CUDA_VISIBLE_DEVICES"]}')
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from omegaconf import OmegaConf

from easyecr.common import common_path
from easyecr.ecr_model.framework.ecr_framework import EcrFramework
from easyecr.ecr_model.model.ecr_model import EcrModel
from easyecr.ecr_model.model.pl_ecr_models.global_local_topic import GlobalLocalTopicModel
from easyecr.ecr_model.cluster.cluster_model import EcrConnectedComponent
from easyecr.ecr_model.cluster.cluster_model import EcrAgglomerativeClustering
from easyecr.ecr_data.data_converter.data_converter import SplitDataConverter
from easyecr.ecr_evaluate.ecr_evaluate import Evaluator
from easyecr.utils import object_utils


def get_dataset(
    model_name: str,
    dataset_name: str,
    train_dir: Optional[str] = None,
    dev_dir: Optional[str] = None,
    test_dir: Optional[str] = None,
    total_dir: Optional[str] = None,
    use_cache: bool = True,
    cache_dir: str = "",
):
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        train_ecr_data_cache_path = os.path.join(cache_dir, f"{model_name}-{dataset_name}.train.pkl")
        dev_ecr_data_cache_path = os.path.join(cache_dir, f"{model_name}-{dataset_name}.dev.pkl")
        test_ecr_data_cache_path = os.path.join(cache_dir, f"{model_name}-{dataset_name}.test.pkl")
        if os.path.exists(train_ecr_data_cache_path):
            train_ecr_data = object_utils.load(train_ecr_data_cache_path)
            dev_ecr_data = object_utils.load(dev_ecr_data_cache_path)
            test_ecr_data = object_utils.load(test_ecr_data_cache_path)
        else:
            if dataset_name == "kbpmix":
                train_ecr_data, dev_ecr_data, test_ecr_data = SplitDataConverter().split(
                    dataset_name=dataset_name, train_path=train_dir, dev_path=dev_dir, test_path=test_dir
                )
            elif dataset_name == "ace2005eng":
                train_ecr_data, dev_ecr_data, test_ecr_data = SplitDataConverter().split(
                    dataset_name=dataset_name, total_path=total_dir
                )
            elif dataset_name == "mavenere":
                train_ecr_data, dev_ecr_data, test_ecr_data = SplitDataConverter().split(
                    dataset_name=dataset_name, train_path=train_dir, dev_path=dev_dir, test_path=test_dir
                )
            else:
                raise NotImplementedError

            object_utils.save(train_ecr_data, train_ecr_data_cache_path)
            object_utils.save(dev_ecr_data, dev_ecr_data_cache_path)
            object_utils.save(test_ecr_data, test_ecr_data_cache_path)
    else:
        if dataset_name == "kbpmix":
            train_ecr_data, dev_ecr_data, test_ecr_data = SplitDataConverter().split(
                dataset_name=dataset_name, train_path=train_dir, dev_path=dev_dir, test_path=test_dir
            )
        elif dataset_name == "ace2005eng":
            train_ecr_data, dev_ecr_data, test_ecr_data = SplitDataConverter().split(
                dataset_name=dataset_name, total_path=total_dir
            )
        elif dataset_name == "mavenere":
            train_ecr_data, dev_ecr_data, test_ecr_data = SplitDataConverter().split(
                dataset_name=dataset_name, train_path=train_dir, dev_path=dev_dir, test_path=test_dir
            )
        else:
            raise NotImplementedError

    return train_ecr_data, dev_ecr_data, test_ecr_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_filename",
        type=str,
        default="global_local_topic_kbpmix",
        choices=[
            "global_local_topic_mavenere",
            "global_local_topic_ace2005eng",
            "global_local_topic_kbpmix",
            "end_to_end_ecbplus",
            "end_to_end_wec",
            "contrastive_representation_learning_ideacar",
            "crl_ecbplus",
        ],
    )
    args = parser.parse_args()

    config_filename = args.config_filename
    config_filepath = os.path.join(common_path.project_dir, f"config/model_dataset/{config_filename}.yaml")
    config = OmegaConf.load(config_filepath)
    config = OmegaConf.to_container(config, resolve=True)

    dataset_name = config["ecr_data"]["dataset_name"]
    if "ace" not in dataset_name:
        train_data, dev_data, test_data = get_dataset(
            model_name=config_filename,
            dataset_name=dataset_name,
            train_dir=config["ecr_data"]["train_path"],
            dev_dir=config["ecr_data"]["dev_path"],
            test_dir=config["ecr_data"]["test_path"],
            use_cache=True,
            cache_dir=config["common"]["cache_dir"],
        )
    else:
        train_data, dev_data, test_data = get_dataset(
            model_name=config_filename,
            dataset_name=dataset_name,
            total_dir=config["ecr_data"]["total_path"],
            use_cache=True,
            cache_dir=config["common"]["cache_dir"],
        )
    print("Split data load")

    mode = config["common"]["mode"]
    ecr_model: EcrModel = GlobalLocalTopicModel(
        mode=mode,
        model_dir=config["EcrFramework1"]["parameters"]["ecr_model"]["parameters"]["common"]["model_dir"],
        model_filename=config_filename,
        trainer_parameters=config["EcrFramework1"]["parameters"]["ecr_model"]["parameters"]["trainer_parameters"],
        conf=config["EcrFramework1"]["parameters"]["ecr_model"]["parameters"],
    )

    cluster_model = EcrConnectedComponent(**config["EcrFramework1"]["parameters"]["cluster_model"]["parameters"])

    evaluator = Evaluator(**config["EcrFramework1"]["parameters"]["evaluator"]["parameters"])

    framework = EcrFramework(
        predict_topic=config["EcrFramework1"]["parameters"]["predict_topic"],
        evaluate_topic=config["EcrFramework1"]["parameters"]["evaluate_topic"],
        main_metric=config["EcrFramework1"]["parameters"]["main_metric"],
        ecr_model=ecr_model,
        ecr_model_output_tag=config["EcrFramework1"]["parameters"]["ecr_model_output_tag"],
        cluster_model=cluster_model,
        evaluator=evaluator,
    )

    if mode == "train":
        framework.train(train_data, dev_data)
    elif mode == "evaluate":
        metrics = framework.evaluate(test_data)
        print(metrics)
    elif mode == "predict":
        result = framework.predict(test_data, output_tag="event_id_pred")
        print(result)
    else:
        raise NotImplementedError(mode)
    print("end")
