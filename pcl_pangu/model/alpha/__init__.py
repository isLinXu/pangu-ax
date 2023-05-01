#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Date: 2022/7/20
# @Author: 2022 PCL

from .alpha import train, fine_tune, inference
from .config_alpha import model_config_gpu, \
    model_config_npu, \
    model_config_onnx, \
    MODEL_CONFIG

