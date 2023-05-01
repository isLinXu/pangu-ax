#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Date: 2022/8/16
# @Author: pcl
import time
import requests
from pcl_pangu.online.infer.pangu_alpha_dto import reset_default_response, \
    send_requests_pangu_alpha, send_requests_pangu_alpha_old, get_response
from pcl_pangu.online.infer.pangu_evolution_dto import PanguEvolutionDTO

def ErrorMessageConverter(result_response):
    ErrorMessages = ['Wating for reply TimeoutError', '当前排队人数过多，请稍后再点击！']
    WarningMessages = ['OutputEmptyWarning']
    if result_response["results"]["generate_text"] in ErrorMessages:
        result_response["results"]["generate_text"] = ''
        result_response['status'] = False

    if result_response["results"]["generate_text"] in WarningMessages:
        result_response["results"]["generate_text"] = ''
        result_response['status'] = True
    return result_response

class Infer(object):

    def __init__(self):
        pass

    @classmethod
    def do_generate_pangu_alpha(cls, model, prompt_input, api_key=None, max_token=None, top_k=None, top_p=None, **kwargs):
        payload = {
            'api_key': api_key,
            'u': prompt_input,
            'top_k': top_k,
            'top_p': top_p,
            'result_len': max_token,
            # 'isWaiting': 'false'
        }
        reset_default_response()

        send_requests_pangu_alpha_old(payload)
        result_response = get_response()
        result_response['api_key'] = api_key
        result_response['model'] = model

        return ErrorMessageConverter(result_response)

    @classmethod
    def do_generate_pangu_evolution(cls, model, prompt_input, api_key=None, max_token=None, top_k=None, top_p=None, **kwargs):

        request = PanguEvolutionDTO.build_request(model, api_key, prompt_input, max_token, top_k, top_p)
        default_response = PanguEvolutionDTO.build_default_response(api_key, model, prompt_input)
        # response = ErrorMessageConverter(PanguEvolutionDTO.do_remote_infer_v1(request, default_response))
        response = PanguEvolutionDTO.do_remote_infer_v2(request, default_response)
        return response

    @classmethod
    def generate(cls, model, prompt_input, api_key, max_token=None, top_k=None, top_p=None, **kwargs):
        """
        model: 模型
        prompt_input: 文本输入，可以结合prompt做为整体输入
        max_token:
        top_k: 随机采样参数
        top_p: 随机采样参数
        kwargs: 不同模型支持的其他参数
        """
        if "pangu-alpha-13B-md"==model:
            return cls.do_generate_pangu_alpha(model, prompt_input, api_key, max_token, top_k, top_p, **kwargs)

        elif "pangu-alpha-evolution-2B6-pt"==model:
            return cls.do_generate_pangu_evolution(model, prompt_input, api_key, max_token, top_k, top_p, **kwargs)

        elif "chat-pangu"==model:
            return cls.do_generate_pangu_evolution(model, prompt_input, api_key, max_token, top_k, top_p, **kwargs)

        else:
            defalut_response = {"status": "The model does not exist."}
            print("Error model.")
            return defalut_response
