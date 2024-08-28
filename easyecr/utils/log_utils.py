#!/usr/bin/env python
# -*- coding:utf-8 -*-　
"""
# https://docs.python.org/zh-cn/3/howto/logging.html#logging-basic-tutorial


"""

import logging
import sys


def get_logger(logger_name: str, log_level=logging.INFO,
               format: str = '%(asctime)s - [%(name)s:%(lineno)d] - %(levelname)s - %(message)s'):
    """

    :param logger_name:
    :param log_level:
    :param format:
    :return:
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    logger.propagate = False

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(log_level)
    formatter = logging.Formatter(format)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    return logger
