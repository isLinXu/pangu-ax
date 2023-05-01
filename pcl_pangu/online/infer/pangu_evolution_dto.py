#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Date: 2022/8/17
# @Author: pcl
import time
import json
import requests


class PanguEvolutionDTO(object):

    pangu_evolution_url_v1 = "https://pangu-alpha.pcl.ac.cn/query_advanced?"
    pangu_evolution_url_v2 = "https://pangu-alpha.pcl.ac.cn/query_advanced_api"

    DEFAULT_MAX_TOKEN = 50
    TOP_P = 0.0
    TOP_K = 1

    def __init__(self):
        pass

    @classmethod
    def build_request(cls, model, api_key, prompt_input, max_token, top_k, top_p):
        max_token = cls.DEFAULT_MAX_TOKEN if max_token is None else max_token
        top_k = cls.TOP_K if top_k is None else top_k
        top_p = cls.TOP_P if top_p is None else top_p

        request = {
                   "api_key": api_key,
                   "input_query": prompt_input,
                   "result_len": max_token,
                   "top_k": top_k,
                   "top_p": top_p,
                   "task_name": "",
                   "model": model,
                   }
        return request

    @classmethod
    def build_default_response(cls, api_key, model, prompt_input):
        default_response = {
            "api_key": api_key,
            "api_access_status": True,
            "model": model,
            "object": "generate",
            "results": {
                "prompt_input": prompt_input,
                "generate_text": None,
                "logprobs": None,
            },
            "status": False
        }
        return default_response

    @classmethod
    def do_remote_infer_v1(cls, request, default_response):
        try:
            response = requests.get(cls.pangu_evolution_url_v1, params=request, headers={'Connection': 'close'})
            if response.status_code == 200:
                result = response.json()["rsvp"]
                openi_access_flag = response.json()["openi_access_flag"]
                if result and openi_access_flag:
                    default_response["results"]["generate_text"] = result[-1]
                    default_response["status"] = True
                    return default_response
                elif openi_access_flag is False:
                    default_response["api_access_status"] = False
                    print("Error api_key!")
                    return default_response

        except:
            time.sleep(10)
            print("Connection refused by the server!")

        print("Error response!")
        return default_response

    @classmethod
    def do_remote_infer_v2(cls, request, default_response):
        data = json.dumps(request).encode('utf-8')
        header = {'Connection': 'close'}
        try:
            resp = requests.post(cls.pangu_evolution_url_v2, data=data, headers=header)
            response = json.loads(resp.text)

            status_code = response["code"]
            default_response["results"]["generate_text"] = response["response_info"]["model_response"]
            default_response["status"] = response["response_info"]["status"]
            default_response["api_access_status"] = response["response_info"]["api_access_status"]

        except:
            time.sleep(5)
            print("Connection refused by the server!")

        return default_response