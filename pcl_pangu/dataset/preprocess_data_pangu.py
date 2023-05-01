# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Processing data for pretraining."""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../model/panguAlpha_pytorch'))

import argparse
import multiprocessing
import glob
from megatron.tokenizer.tokenization_jieba import JIEBATokenizer
from pcl_pangu.tokenizer import vocab_4w
import time

import torch
try:
    import nltk
    nltk_available = True
except ImportError:
    nltk_available = False

from megatron.data import indexed_dataset



class IdentitySplitter(object):
    def tokenize(self, *text):
        return text


class Encoder(object):
    def __init__(self, args):
        self.args = args
        if self.args.vocab_file == vocab_4w:
            self.tokenizer = JIEBATokenizer(self.args.vocab_file)
        else:
            print("> We recommend you using JIEBATokenizer to build your vocab ")
            self.tokenizer = JIEBATokenizer(self.args.vocab_file)

    def encode(self, iterator):
        key = self.args.json_keys[0]
        len_paras = 0
        ids = {}
        doc_ids = []
        
        encode_start_time = time.time()
        file_num = 0
        for file_path in iterator:
            print(file_path)
            each_start_time = time.time()
            json_line = open(file_path, 'r', encoding='utf-8')
            strr = json_line.read()
            lista = strr.split('\n\n')
            len_paras += len(lista)
            for para in lista:
                if para:
                    contenta = self.tokenizer.tokenize(para)
                    para_ids = self.tokenizer.convert_tokens_to_ids(contenta)
                    if len(para_ids) > 0:
                        doc_ids.append(para_ids)
                        if self.args.append_eod:
                            for i in range(self.args.eod_num):
                                doc_ids[-1].append(self.tokenizer.eod_id)
                    # print(doc_ids)
            each_end_time = time.time()
            print("encode this file using {}s".format(each_end_time - each_start_time))
        ids[key] = doc_ids
        encode_end_time = time.time()
        print("FINISHING ENCODING, USING {}s".format(encode_end_time - encode_start_time))
        
        return ids, len_paras
        # print('len_paras',len_paras)


def package_file(it, n):
    """ package multiple files"""
    stop = False
    while not stop:
        batch = []
        for _ in range(n):
            try:
                batch.append(next(it))
            except StopIteration:
                stop = True
        if not batch:
            break
        yield batch
    
    
def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, default='/raid/gpt3-train-data/data-v1/new2016zh/txt-data/train/0000*.txt',
                       help='Path to input txt')
    group.add_argument('--json-keys', nargs='+', default=['text'],
                       help='space separate listed of keys to extract from json')
    group.add_argument('--split-sentences', action='store_true',
                       help='Split documents into sentences.')

    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--vocab-file', type=str, default='tokenizer/bpe_4w_pcl/vocab',
                       help='Path to the vocab file')
    group.add_argument('--append-eod', action='store_true', default=True,
                       help='Append an <eod> token to the end of a document.')
    group.add_argument('--eod-num', type=int, default=1,
                       help='eot number.')

    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, default="",
                       help='Path to binary output file without suffix')
    group.add_argument('--dataset-impl', type=str, default='mmap',
                       choices=['lazy', 'cached', 'mmap'])

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, default=200,
                       help='Number of worker processes to launch')
    group.add_argument('--log-interval', type=int, default=1,
                       help='Interval between progress updates')
    args = parser.parse_args()
    args.keep_empty = False

    return args