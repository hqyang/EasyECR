#!/usr/bin/env python
# -*- coding:utf-8 -*-　
"""

nohup sh run.sh 3 easyecr/pipeline/bootstrap_v4.py --config_filename contrastive_representation_learning_ideacar > bootstrap.log 2>&1 &

nohup sh run.sh 5 easyecr/pipeline/bootstrap_v4.py --config_filename end_to_end_ecbplus > end_to_end_ecbplus.log 2>&1 &

nohup sh run.sh 1 easyecr/pipeline/bootstrap_v4.py --config_filename end_to_end_wec > end_to_end_wec.log 2>&1 &

nohup sh run.sh 5 easyecr/pipeline/bootstrap_v4.py --config_filename crl_ecbplus > crl_ecbplus.log 2>&1 &



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
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = 'false'

from omegaconf import OmegaConf

from easyecr.common import common_path
from easyecr.ecr_model.framework.ecr_framework import EcrFramework
from easyecr.ecr_model.model.ecr_model import EcrModel
from easyecr.ecr_model.model.pl_ecr_models.contrastive_representation_learning \
    import ContrastiveRepresentationLearningEcrModel
from easyecr.ecr_model.model.pl_ecr_models.end_to_end import EndToEndEcrModel
from easyecr.ecr_model.cluster.cluster_model import EcrAgglomerativeClustering
from easyecr.ecr_data.data_converter.data_converter import DataConverter
from easyecr.ecr_evaluate.ecr_evaluate import Evaluator
from easyecr.ecr_model.ecr_tagger.ecr_tagger import EcrTagger
from easyecr.ecr_model.ecr_tagger.contextualized_representation_tagger import ContextualizedRepresentationTagger
from easyecr.utils import object_utils
from easyecr.ecr_data.datasets.wec.wec_eng import WECEngDataset


def get_dataset(model_name: str, dataset_name: str, train_dir: str, dev_dir: str, test_dir: str,
                ecr_tagger: Optional[EcrTagger] = None,
                use_cache: bool = True,
                cache_dir: str = ''):
    """

    Args:
        model_name:
        dataset_name:
        train_dir:
        dev_dir:
        test_dir:
        ecr_tagger:
        use_cache:
        cache_dir:

    Returns:

    """
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        train_ecr_data_cache_path = os.path.join(cache_dir,
                                                 f'{model_name}-{dataset_name}.train.pkl')
        dev_ecr_data_cache_path = os.path.join(cache_dir,
                                               f'{model_name}-{dataset_name}.dev.pkl')
        test_ecr_data_cache_path = os.path.join(cache_dir,
                                                f'{model_name}-{dataset_name}.test.pkl')
        if os.path.exists(train_ecr_data_cache_path):
            train_ecr_data = object_utils.load(train_ecr_data_cache_path)
            dev_ecr_data = object_utils.load(dev_ecr_data_cache_path)
            test_ecr_data = object_utils.load(test_ecr_data_cache_path)
        else:
            if dataset_name == 'wec-eng':
                train_ecr_data = WECEngDataset(directory=train_dir, selection='Train').to_ecr_data()
                dev_ecr_data = WECEngDataset(directory=dev_dir, selection='Dev').to_ecr_data()
                test_ecr_data = WECEngDataset(directory=test_dir, selection='Test').to_ecr_data()
            else:
                train_ecr_data = DataConverter.from_directory(dataset_name, train_dir)
                dev_ecr_data = DataConverter.from_directory(dataset_name, dev_dir)
                test_ecr_data = DataConverter.from_directory(dataset_name, test_dir)

            if ecr_tagger:
                train_ecr_data = ecr_tagger.predict(train_ecr_data, output_tag='mention')
                dev_ecr_data = ecr_tagger.predict(dev_ecr_data, output_tag='mention')
                test_ecr_data = ecr_tagger.predict(test_ecr_data, output_tag='mention')

            object_utils.save(train_ecr_data, train_ecr_data_cache_path)
            object_utils.save(dev_ecr_data, dev_ecr_data_cache_path)
            object_utils.save(test_ecr_data, test_ecr_data_cache_path)
    else:
        if dataset_name == 'wec-eng':
            train_ecr_data = WECEngDataset(directory=train_dir, selection='Train').to_ecr_data()
            dev_ecr_data = WECEngDataset(directory=dev_dir, selection='Dev').to_ecr_data()
            test_ecr_data = WECEngDataset(directory=test_dir, selection='Test').to_ecr_data()
        else:
            train_ecr_data = DataConverter.from_directory(dataset_name, train_dir)
            dev_ecr_data = DataConverter.from_directory(dataset_name, dev_dir)
            test_ecr_data = DataConverter.from_directory(dataset_name, test_dir)

        if ecr_tagger:
            train_ecr_data = ecr_tagger.predict(train_ecr_data, output_tag='mention')
            dev_ecr_data = ecr_tagger.predict(dev_ecr_data, output_tag='mention')
            test_ecr_data = ecr_tagger.predict(test_ecr_data, output_tag='mention')

    return train_ecr_data, dev_ecr_data, test_ecr_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_filename", default='crl_ecbplus', type=str,
                        choices=['end_to_end_ecbplus', 'end_to_end_wec','contrastive_representation_learning_ideacar', 'crl_ecbplus'])
    args = parser.parse_args()

    config_filename = args.config_filename
    config_filepath = os.path.join(common_path.project_dir, f'config/model_dataset/{config_filename}.yaml')
    config = OmegaConf.load(config_filepath)

    dataset_name = config['ecr_data']['dataset_name']
    if 'tagger' in config:
        ecr_tagger: EcrTagger = ContextualizedRepresentationTagger(config['tagger']['transformer_model'])
    else:
        ecr_tagger: EcrTagger = None

    train_data, dev_data, test_data = get_dataset(
        config['ecr_model']['model_name'], dataset_name,
        config['ecr_data']['dataset_train_dir'],
        config['ecr_data']['dataset_dev_dir'],
        config['ecr_data']['dataset_test_dir'],
        ecr_tagger,
        use_cache=True,
        cache_dir=config['common']['cache_dir']
    )

    if 'crl' in config_filename:
        model: EcrModel = ContrastiveRepresentationLearningEcrModel(conf=config)
    else:
        model: EcrModel = EndToEndEcrModel(conf=config)

    cluster_model = EcrAgglomerativeClustering(**config['cluster_model']['parameters'])

    evaluator = Evaluator(**config['evaluator'])

    framework = EcrFramework(config, model, cluster_model, evaluator)

    mode = config['common']['mode']
    if mode == 'train':
        framework.train(train_data, dev_data)
    elif mode == 'evaluate':
        metrics = framework.evaluate(test_data)
        print(metrics)
    elif mode == 'predict':
        result = framework.predict(test_data, output_tag='event_id_pred')
    else:
        raise NotImplementedError(mode)

    print('end')
