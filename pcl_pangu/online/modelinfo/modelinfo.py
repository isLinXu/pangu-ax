#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Date: 2022/8/12
# @Author: pcl
from loguru import logger


class ModelInfo(object):

    models = {
        "data": [
            {
                "id": "pangu-alpha-13B-2",
                "model": "pangu-alpha-13B-md",
                "object": "model",
                "owned_by": "pcl",
                "permission": ["generate"]
            },
            {
                "id": "pangu-alpha-evolution-1",
                "model": "pangu-alpha-evolution-2B6-pt",
                "object": "model",
                "owned_by": "pcl",
                "permission": ["generate"]
            },
            ],
        "object": "list"
        }

    @classmethod
    def model_list(cls, api_key=None):
        return cls.models

    @classmethod
    def model_info(cls, model, api_key=None):
        defalut_response = {"status": "The model does not exist."}

        for cur_model in cls.models["data"]:
            if model == cur_model["model"]:
                return cur_model
        logger.warning("Wrong model: {}".format(model))

        return defalut_response