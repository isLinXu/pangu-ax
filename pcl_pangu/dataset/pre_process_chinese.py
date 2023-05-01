# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""
transform wikitext-2, wikitext-103, lambada, openwebtext dataset to mindrecord.
"""
import argparse
import json
import os
import re
import sys
import random

import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../model/panguAlpha_mindspore'))
from pcl_pangu.model.panguAlpha_mindspore.src.tokenization_jieba import JIEBATokenizer
from pcl_pangu.tokenizer import vocab_4w, vocab_13w

try:
    import moxing as mox
    modelarts_flag = True
except:
    ImportWarning("> Not using [PCL-yunnao-modelarts] resources! Are u using local NPU-machine?")
    modelarts_flag = False

from mindspore.mindrecord import FileWriter
from multiprocessing import current_process

tokenizer = None
writer = None

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_type', type=str, default='openwebtext')
parser.add_argument('--data_url', type=str, default='openwebtext')
parser.add_argument('--train_url', type=str, default='openwebtext')
###########################
parser.add_argument('--input_glob', type=str, default='/userhome/dataset/chinese_txt/webtext2019zh/*.txt')
#############################
parser.add_argument('--output_dir', type=str,
                    default='/cache/dataMindrecord/PD_NewPET_0704_mindrecord')
parser.add_argument('--file_partition', type=int, default=1)
parser.add_argument('--file_batch_size', type=int, default=1) #不影响结果
parser.add_argument('--num_process', type=int, default=200)
parser.add_argument('--tokenizer', type=str, default='vocab_4w')
parser.add_argument('--SEQ_LEN', type=int, default=1025)
parser.add_argument('--rankOfCluster', type=str, default='0of1')

args = parser.parse_args()

SEQ_LEN = args.SEQ_LEN  # the length of sample

# EOT = 50256  # id of endoftext
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def setup_tokenizer():
    global tokenizer
    if args.tokenizer == 'vocab_4w':
        vocab_path = vocab_4w + '.vocab'
        model_file = vocab_4w + '.model'
        tokenizer = JIEBATokenizer(vocab_path, model_file)
    elif args.tokenizer == 'vocab_13w':
        from pcl_pangu.tokenizer.spm_13w.tokenizer import SpmTokenizer, langs_ID, translate_ID
        vocab_file = vocab_13w
        tokenizer = SpmTokenizer(vocab_file)
    else:
        vocab_path = vocab_4w + '.vocab'
        model_file = vocab_4w + '.model'
        tokenizer = JIEBATokenizer(vocab_path, model_file)
    return tokenizer

def setup_writer(args):
    global writer
    schema = {"input_ids": {"type": "int32", "shape": [-1]}, }
    writer = FileWriter(file_name=args.output_dir,
                        shard_num=args.file_partition)
    writer.add_schema(schema, args.dataset_type)
    writer.open_and_set_header()
    return writer


def chunks(lst, n):
    """ yield n sized chunks from list"""
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

# 将一个list尽量均分成n份，限制len(list)==n，份数大于原list内元素个数则分配空list[]
def divideIntoNstrand(listTemp, n):
    twoList = [ [] for i in range(n)]
    for i,e in enumerate(listTemp):
        twoList[i%n].append(e)
    return twoList


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


def clean_wikitext(string):
    """ cleaning wikitext dataset"""
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" "+chr(176)+" ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    return string


def padding_eot(chunk):
    pad = [tokenizer.pad_id] * (SEQ_LEN - len(chunk))
    chunk.extend(pad)
    return chunk


def tokenize_openwebtext(iterator):
    """ tokenize openwebtext dataset"""
    content = []
    for file_path in iterator:
        if os.path.getsize(file_path) == 0:
            continue

        content = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for para in f.read().split("\n\n"):
                if para:
                    ###############################################
                    tokenized_text = tokenizer.tokenize(para)
                    content += tokenizer.convert_tokens_to_ids(tokenized_text) + [tokenizer.eot_id]
                    ###########################################

        for chunk in chunks(content, SEQ_LEN):
            sample = {}
            if len(chunk) == SEQ_LEN:
                sample['input_ids'] = np.array(chunk, dtype=np.int32)
                yield sample
            else:
                sample['input_ids'] = np.array(padding_eot(chunk), dtype=np.int32)
                yield sample


def tokenize_openwebtext_mpangu(iterator):
    """ tokenize openwebtext dataset"""
    from pcl_pangu.tokenizer.spm_13w.tokenizer import langs_ID, translate_ID
    for file_path in iterator:
        if os.path.getsize(file_path) == 0:
            continue

        content = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for para in f.read().split("\n\n"):
                if para:
                    ###############################################
                    ##----------------------单语语料处理--------------------
                    if 'corpus' not in file_path.split('/')[-1]:
                        langs = file_path.split('/')[-1][:2]
                        id_langs = langs_ID[langs]

                        para = '' + para
                        tokenized_text_langs = tokenizer.tokenize(para)
                        langs_id = tokenizer.convert_tokens_to_ids(tokenized_text_langs)
                        content.append([id_langs] + langs_id + [tokenizer.eot_id])
                    else:
                        ##----------------双语语料处理------------------------------
                        src_langs = file_path.split('/')[-1][:2]
                        tag_langs = file_path.split('/')[-1][3:5]

                        try:
                            src_data, tag_data = para.split("\t")
                            # pangu 4w词表中文文本需要jiaba分词，spm中文文本不需要jieba分词
                            src_data_new = '' + src_data
                            tag_data_new = '' + tag_data

                            # zh corpus save ,other corpus save mono random
                            if isinstance(src_data_new, str) and isinstance(tag_data_new, str):
                                tokenized_text_src = tokenizer.tokenize(src_data_new)
                                src_id = tokenizer.convert_tokens_to_ids(tokenized_text_src)

                                tokenized_text_tag = tokenizer.tokenize(tag_data_new)
                                tag_id = tokenizer.convert_tokens_to_ids(tokenized_text_tag)

                                content.append([langs_ID[src_langs]] + \
                                               src_id + \
                                               [translate_ID] + \
                                               [langs_ID[tag_langs]] + \
                                               tag_id + [tokenizer.eot_id])
                                content.append([langs_ID[tag_langs]] + \
                                               tag_id + \
                                               [translate_ID] + \
                                               [langs_ID[src_langs]] + \
                                               src_id + [tokenizer.eot_id])
                            else:
                                print("Not 2 para str...\n")
                        except:
                            print("Split error, jump...", para)
                    ###########################################

        random.shuffle(content)
        content_new = []
        for i in content:
            content_new += i

        for chunk in chunks(content, SEQ_LEN):
            sample = {}
            if len(chunk) == SEQ_LEN:
                sample['input_ids'] = np.array(chunk, dtype=np.int32)
                yield sample
            else:
                sample['input_ids'] = np.array(padding_eot(chunk), dtype=np.int32)
                yield sample


def tokenize_openwebtext_padEachPara(iterator):
    """ tokenize openwebtext dataset"""
    content = []
    for file_path in iterator:
        if os.path.getsize(file_path) == 0:
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            for para in f.read().split("\n\n"):
                if para:
                    content = []
                    ###############################################
                    tokenized_text = tokenizer.tokenize(para)
                    content += tokenizer.convert_tokens_to_ids(tokenized_text) + [tokenizer.eot_id]
                    ###########################################

                    for chunk in chunks(content, SEQ_LEN):
                        sample = {}
                        if len(chunk) == SEQ_LEN:
                            sample['input_ids'] = np.array(chunk, dtype=np.int32)
                            yield sample
                        else:
                            sample['input_ids'] = np.array(padding_eot(chunk), dtype=np.int32)
                            yield sample


def tokenize_wiki(file_path):
    """tokenize wikitext-2/wikitext-103 dataset"""
    content = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for para in clean_wikitext(f.read()).split("\n\n"):
            if para and para.strip().startswith('=') is False:
                tokenized_text = tokenizer.tokenize(para)
                content += tokenizer.convert_tokens_to_ids(tokenized_text) + [
                    tokenizer.eot_id]
    for chunk in chunks(content, SEQ_LEN):
        sample = {}
        if len(chunk) == SEQ_LEN:
            sample['input_ids'] = np.array(chunk, dtype=np.int32)
            yield sample


def tokenize_lambada(file_path):
    """tokenize lambada dataset"""
    content = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            para = json.loads(line)['text'].replace(
                "“", '""').replace("”", '"').strip().strip(".")
            tokenized_text = tokenizer.tokenize(para)
            content += tokenizer.convert_tokens_to_ids(tokenized_text) + [tokenizer.eot_id]
    for chunk in chunks(content, SEQ_LEN):
        sample = {}
        if len(chunk) == SEQ_LEN:
            sample['input_ids'] = np.array(chunk, dtype=np.int32)
            yield sample

def conver_words_to_ids_ch(words, word2id, numchword):
    ids = []
    for word in words:
        if word in word2id:
            ids.append(word2id[word])
    return ids


def task_unit(iterator, parallel_writer=True):
    """task for each process"""
    p = current_process()
    index = p.pid if p.pid else 0

    item_iter = tokenize_openwebtext(iterator)
    batch_size = 1024  # size of write batch
    count = 0
    while True:
        data_batch = []
        try:
            for _ in range(batch_size):
                data_batch.append(next(item_iter))
                count += 1
            writer.write_raw_data(data_batch, parallel_writer=parallel_writer)
            print("Process {} transformed {} records.".format(
                index, count))
        except StopIteration:
            if data_batch:
                writer.write_raw_data(data_batch,
                                      parallel_writer=parallel_writer)
                print("Process {} transformed {} records.".format(
                    index, count))
            break


def task_unit_mPangu(iterator, parallel_writer=True):
    """task for each process"""
    p = current_process()
    index = p.pid if p.pid else 0

    item_iter = tokenize_openwebtext_mpangu(iterator)
    batch_size = 1024  # size of write batch
    count = 0
    while True:
        data_batch = []
        try:
            for _ in range(batch_size):
                data_batch.append(next(item_iter))
                count += 1
            writer.write_raw_data(data_batch, parallel_writer=parallel_writer)
            print("Process {} transformed {} records.".format(
                index, count))
        except StopIteration:
            if data_batch:
                writer.write_raw_data(data_batch,
                                      parallel_writer=parallel_writer)
                print("Process {} transformed {} records.".format(
                    index, count))
            break