#!/usr/bin/env python
# -*- coding:utf-8 -*-　
"""

nohup sh run.sh 3 easyecr/pipeline/cluster_ecr_model_pipeline.py --config_filename cluster_ecr_model_ecbplus > cluster_ecr_model_ecbplus.log 2>&1 &



Date: 2023/11/28 11:01
"""

from typing import Optional
import argparse
import os
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    print(f'set CUDA_VISIBLE_DEVICES: {os.environ["CUDA_VISIBLE_DEVICES"]}')
else:
    print(f'CUDA_VISIBLE_DEVICES: {os.environ["CUDA_VISIBLE_DEVICES"]}')
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = 'false'

from omegaconf import OmegaConf

from easyecr.common import common_path
from easyecr.ecr_model.framework.ecr_framework import EcrFramework
from easyecr.ecr_model.model.ecr_model import EcrModel
from easyecr.ecr_model.model.pl_ecr_models.lemma_distance_ecr_model import LemmaDistanceEcrModel
from easyecr.ecr_model.model.pl_ecr_models.cluster_ecr_model import ClusterEcrModel
from easyecr.ecr_model.cluster.cluster_model import EcrAgglomerativeClustering
from easyecr.ecr_evaluate.ecr_evaluate import Evaluator
from easyecr.ecr_model.ecr_tagger.ecr_tagger import EcrTagger
from easyecr.utils import object_utils
from easyecr.ecr_model.model.pl_ecr_models.discourse_coherence_theory_ecr_model import DiscourseCoherenceTheoryEcrModel
from easyecr.ecr_model.ecr_tagger.doc_tagger import DocTagger
from easyecr.ecr_data.data_converter.data_converter import SplitDataConverter
from easyecr.ecr_model.model.simple_ecr_models.embedding_distance import EmbeddingDistance
from easyecr.utils import load_data

def get_dataset(config_name: str,
                cache_dir: str,
                dataset_name: str,
                debug: bool = False,
                config = None,
                raw_data_cache_dir: str = '/home/nobody/project/EasyECR/cache',
                train_path: str = '',
                dev_path: str = '',
                test_path: str = '',
                total_path: str = ''
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
    train_data_cache_path = os.path.join(cache_dir, f'{config_name}.train.pkl')
    dev_data_cache_path = os.path.join(cache_dir, f'{config_name}.dev.pkl')
    test_data_cache_path = os.path.join(cache_dir, f'{config_name}.test.pkl')
    if os.path.exists(train_data_cache_path) and os.path.exists(dev_data_cache_path) \
            and os.path.exists(test_data_cache_path):
        train_data = object_utils.load(train_data_cache_path)
        dev_data = object_utils.load(dev_data_cache_path)
        test_data = object_utils.load(test_data_cache_path)
    else:
        train_data, dev_data, test_data = load_data.get_dataset(dataset_name,
                                                                cache_dir=raw_data_cache_dir,
                                                                train_path=train_path,
                                                                dev_path=dev_path,
                                                                test_path=test_path,
                                                                total_path=total_path)
        object_utils.save(train_data, train_data_cache_path)
        object_utils.save(dev_data, dev_data_cache_path)
        object_utils.save(test_data, test_data_cache_path)

    if debug:
        train_data.reduce_mentions_for_debugging()
        dev_data.reduce_mentions_for_debugging()
        test_data.reduce_mentions_for_debugging()
    return train_data, dev_data, test_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_filename", default='cluster_ecr_model_ecbplus', type=str)
    args = parser.parse_args()

    config_filename = args.config_filename
    config_filepath = os.path.join(common_path.project_dir, f'config/model_dataset/{config_filename}.yaml')
    config = OmegaConf.load(config_filepath)
    mode = config['common']['mode']

    dataset_name = config['ecr_data']['dataset_name']
    train_data, dev_data, test_data = get_dataset(
        config_filename,
        cache_dir=config['common']['cache_dir'],
        dataset_name=dataset_name,
        debug=False,
        config=config,
        total_path=config['ecr_data']['total_path'],
    )

    doc_tagger = DocTagger()
    test_data = doc_tagger.predict(test_data, config['DocTagger']['output_tag'])

    cluster_ecr_model = ClusterEcrModel(
        mode=mode,
        model_dir=config['EcrFramework1']['parameters']['ecr_model']['parameters']['common']['model_dir'],
        model_filename=config_filename,
        trainer_parameters=config['EcrFramework1']['parameters']['ecr_model']['parameters']['trainer_parameters'],
        conf=config['EcrFramework1']['parameters']['ecr_model']['parameters']
    )
    evaluator = Evaluator(**config['EcrFramework1']['parameters']['evaluator']['parameters'])
    simple_cluster_model = EcrAgglomerativeClustering(**config['EcrFramework1']['parameters']['cluster_model']['parameters'])
    framework = EcrFramework(config['EcrFramework1']['parameters']['predict_topic'],
                             config['EcrFramework1']['parameters']['evaluate_topic'],
                             config['EcrFramework1']['parameters']['main_metric'],
                             cluster_ecr_model,
                             config['EcrFramework1']['parameters']['ecr_model_output_tag'],
                             simple_cluster_model, evaluator)
    if mode == 'train':
        framework.train(train_data, dev_data, config['EcrFramework1']['output_tag'])
        metrics = framework.evaluate(test_data, config['EcrFramework1']['output_tag'])
        print(metrics)
    elif mode == 'evaluate':
        metrics = framework.evaluate(test_data, config['EcrFramework1']['output_tag'])
        print(metrics)
    elif mode == 'predict':
        result = framework.predict(test_data, config['EcrFramework1']['output_tag'])
    else:
        raise NotImplementedError(mode)

    print('end')
