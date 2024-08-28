#!/usr/bin/env python
# -*- coding:utf-8 -*-　
"""



Date: 2023/10/27 17:04
"""

import torch

input1 = torch.randn(100, 128)
input2 = torch.randn(100, 128)
output = torch.cosine_similarity(input1, input2, dim=0)
print(output)