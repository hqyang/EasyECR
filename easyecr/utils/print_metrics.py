#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""



Date: 2023/12/18 10:06
"""
import json


def print_metrics_for_table(s):
    """

    Args:
        metrics:

    Returns:

    """
    obj = json.loads(s)
    metric_names = ["muc", "bcub", "ceafe", "lea", "CoNLL"]
    result = []
    for name in metric_names:
        metrics = obj[0]["metrics"][name]
        sub_metric_names = ["precision", "recall", "f1"]
        for sub_metric_name in sub_metric_names:
            if sub_metric_name in metrics:
                result.append("%.5f" % metrics[sub_metric_name])
    print("\t".join(result))


if __name__ == "__main__":
    s = """[{"key": "event_id_pred_ecr_framework2|0.9", "metrics": {"mentions": {"recall": 0.8273092369477911, "precision": 0.9395667046750285, "f1": 0.8798718633208755}, "muc": {"recall": 0.8158765159867696, "precision": 0.8447488584474886, "f1": 0.8300616937745372}, "bcub": {"recall": 0.7289001412821917, "precision": 0.10879579368350432, "f1": 0.18933187109249108}, "ceafe": {"recall": 0.005190362440649975, "precision": 0.46194225721784776, "f1": 0.010265383493729952}, "lea": {"recall": 0.7134285436518284, "precision": 0.10784742506365098, "f1": 0.18737046822422146}, "CoNLL": {"f1": 34.3219649453586}}}]"""
    print_metrics_for_table(s)
