#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Date: 2022/8/17
# @Author: pcl
import requests
import time

default_response = {
    "api_key": None,
    "api_access_status": True,
    "model": None,
    "object": "generate",
    "results": {
        "prompt_input": None,
        "generate_text": None,
        "logprobs": None,
    },
    "status": True
}

def reset_default_response():
    global default_response
    default_response = {
        "api_key": None,
        "api_access_status": True,
        "model": None,
        "object": "generate",
        "results": {
            "prompt_input": None,
            "generate_text": None,
            "logprobs": None,
        },
        "status": True
    }


def auto_payload_topParams(payload):
    if payload['result_len'] is None:
        payload['result_len'] = 50
        
    if payload['top_k'] is None and payload['top_p'] is None:
        payload['top_k'] = 0
        payload['top_p'] = 0.9
    elif payload['top_k'] is None:
        payload['top_k'] = 1
    elif payload['top_p'] is None:
        payload['top_p'] = 0.9
    return payload

def send_requests_pangu_alpha(payload):
    global default_response
    payload = auto_payload_topParams(payload)
    response = requests.get('https://pangu-alpha.pcl.ac.cn/query?', params=payload)
    if response.status_code == 200:
        result = response.json()['rsvp']
        if result is None:
            payload['isWaiting'] = 'true'
            time.sleep(10)
            send_requests_pangu_alpha(payload)
        else:
            default_response['results']['prompt_input'] = payload['u']
            default_response['results']['generate_text'] = result[-1]
    else:
        default_response['status'] = False
        print("Error response! checkout your [url] is 'https://pangu-alpha.pcl.ac.cn/query?'\n")


def send_requests_pangu_alpha_old(payload):
    global default_response
    payload = auto_payload_topParams(payload)
    response = requests.get('https://pangu-alpha.pcl.ac.cn/query?', params=payload)
    if response.status_code == 200:
        result = response.json()['rsvp']
        openi_access_flag = response.json()["openi_access_flag"]
        default_response['results']['prompt_input'] = payload['u']
        default_response['results']['generate_text'] = result[-1]
        if result and openi_access_flag:
            default_response["results"]["generate_text"] = result[-1]
            default_response["status"] = True
        elif openi_access_flag is False:
            default_response["results"]["generate_text"] = None
            default_response["status"] = False
            default_response["api_access_status"] = False
    else:
        default_response['status'] = False
        print("Error response! checkout your [url] is 'https://pangu-alpha.pcl.ac.cn/query?'\n")


def get_response():
    return default_response