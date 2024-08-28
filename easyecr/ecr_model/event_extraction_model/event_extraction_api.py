#!/usr/bin/env python
# -*- coding:utf-8 -*-　
"""



Date: 2023/11/21 17:17
"""
import json
from typing import List
from typing import Dict
import traceback

from easyecr.utils import http_utils
from easyecr.conf_loader import load_conf
from easyecr.utils import log_utils
from easyecr.pipeline import bootstrap

CONF = load_conf.load_event_extraction_service_conf()
logger = log_utils.get_logger(__file__)


def extract(texts: List[Dict[str, str]], max_try_num: int = 3):
    """

    :param texts:
    :param max_try_num:
    :return:
    """
    result = []
    url = CONF['ideacar']['url']
    try_num = 0
    while try_num < max_try_num:
        try:
            try_num += 1
            extraction_result = json.loads(http_utils.post(url, texts).text)
            break
        except:
            logger.error(traceback.format_exc())
            extraction_result = []
    if not extraction_result:
        for text in texts:
            text['mentions'] = []
            result.append(text)
    else:
        for text in extraction_result:
            event_with_mentions = {
                'id': text['id'],
                'text': text['text'],
                'mentions': []
            }
            for mention in text['event_list']:
                internal_mention = {}
                internal_mention['doc_id'] = text['id']
                internal_mention['event_type'] = mention['event_type']
                arguments = mention['arguments']
                for argument in arguments:
                    internal_argument = {
                        'text': argument['argument'],
                        'start': argument['argument_start_index'],
                        'end': argument['argument_start_index'] + len(argument['argument']) - 1
                    }
                    role = argument['role']
                    internal_mention[role] = internal_argument
                event_with_mentions['mentions'].append(internal_mention)
            result.append(event_with_mentions)

    return result


if __name__ == '__main__':
    texts = [
        {"id": "value1", "text": "此前大众汽车联合小鹏进行深度绑定研发电动新车，以及奥迪与上汽深化在新能源和智能网联方面的合作就 是最典型的代表。 与此同时，不久前刚刚拿到 福特电马销售权的长安福特，也开始利用中国品牌的技术优势，为自身的新能源转型增添新的砝码。"}
    ]
    result = extract(texts)
    ecr_data = bootstrap.Predictor.to_ecr_data(result)
    print(result)
