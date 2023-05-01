#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Date: 2022/7/20
# @Author: 2022 PCL
import copy
from pcl_pangu.tokenizer import vocab_4w, vocab_13w

MODEL_CONFIG = {
    '350M': {
        'num_layers': 23,
        'hidden_size': 1024,
        'num_attention_heads': 16,
        'seq_length': 1024,
        'max_position_embeddings': 1024,
        'model_parallel_size': 1,
    },
    '2B6': {
        'num_layers': 31,
        'hidden_size': 2560,
        'num_attention_heads': 32,
        'seq_length': 1024,
        'max_position_embeddings': 1024,
        'model_parallel_size': 2,    # 2B6 mp_size >= 2 , recommend for [2,4,8]
    },
    '13B': {
        'num_layers': 39,
        'hidden_size': 5120,
        'num_attention_heads': 40,
        'seq_length': 1024,
        'max_position_embeddings': 1024,
        'model_parallel_size': 8,    # 13B mp_size >= 8 , recommend for [8]
    }
}

DISTRUBUTED_CONFIG = {
    'nnodes':1,
    'node_rank':0,
    'nproc_per_node':1,
    'master_addr':"localhost",
    'master_port':29502
}

vocab_dir = vocab_13w

DEFAULT_CONFIG = {
    'model_parallel_size': 1,
    'batch_size': 8,
    'train_iters': 500000,
    'lr_decay_iters': 320000,
    'save': '',
    'load': '',
    'data_path': '',
    'vocab_file': vocab_dir,
    'merge_file': 'gpt2-merges.txt',
    'data_impl': 'mmap',
    'split': '949,50,1',
    'distributed_backend': 'nccl',
    'lr': 0.00015,
    'lr_decay_style': 'cosine',
    'min_lr': 1.0e-5,
    'weight_decay': 1e-2,
    'clip_grad': 1.0,
    'warmup': 0.01,
    'checkpoint_activations': True,
    'log_interval': 100,
    'save_interval': 1000,
    'eval_interval': 1000,
    'eval_iters': 10,
    'fp16': True,
    'finetune': False,
    'tokenizer_type': 'GPT2BPETokenizer'
}


class model_config_gpu():
    def __init__(self, model='350M',
                 model_parallel_size=1,
                 batch_size=8,
                 train_iters=10000,
                 lr=0.00015,
                 data_path='data',
                 vocab_file=vocab_13w,
                 load=None,
                 save=None,
                 nnodes=1,
                 node_rank=0,
                 nproc_per_node=1,
                 master_addr="localhost",
                 master_port=29500):
        self.model = model
        self.model_parallel_size = model_parallel_size
        self.batch_size = batch_size
        self.train_iters = train_iters
        self.lr_decay_iters = int(0.64 * self.train_iters)
        self.lr = lr
        self.load = load
        if load is not None:
            self.save = self.load if save is None else save
        else:
            self.save = save
        self.data_path = data_path
        self.vocab_file = vocab_file

        DISTRUBUTED_CONFIG['nnodes'] = nnodes
        DISTRUBUTED_CONFIG['node_rank'] = node_rank
        DISTRUBUTED_CONFIG['nproc_per_node'] = nproc_per_node
        DISTRUBUTED_CONFIG['master_addr'] = master_addr
        DISTRUBUTED_CONFIG['master_port'] = master_port

    @staticmethod
    def _dict_to_cmd(config_dict):
        cmd = []
        for k, v in config_dict.items():
            if k == 'model':
                pass
            else:
                if v is True:
                    cmd.append('--' + k.replace('_','-'))
                elif v is False:
                    continue
                else:
                    cmd.append('--{}={}'.format(k.replace('_', '-'), v))
        return cmd

    def _get_training_script_args(self, oneCardInference=False):
        global MODEL_CONFIG, DEFAULT_CONFIG
        if oneCardInference:
            tmp_config = MODEL_CONFIG[self.model]
            tmp_config['model_parallel_size'] = 1
            assert self.model_parallel_size == 1, "> mp=1 when Your using OneCardInference!"
        else:
            tmp_config = MODEL_CONFIG[self.model]
            if self.model_parallel_size == 1:
                self.model_parallel_size = tmp_config['model_parallel_size']
        default_config = copy.deepcopy({**tmp_config, **DEFAULT_CONFIG})
        _vars = vars(self)
        default_config['load'] = self.load
        default_config['save'] = self.save
        default_config['data_path'] = self.data_path
        default_config['vocab_file'] = self.vocab_file
        default_config['lr_decay_iters'] = self.lr_decay_iters
        # _vars.pop('checkpoint_path')
        for k, v in _vars.items():
            default_config[k] = v
        cmd = self._dict_to_cmd(default_config)
        return cmd


class model_config_npu():
    def __init__(self, model='350M',
                 model_parallel_size=1,
                 batch_size=8,
                 train_iters=50000,
                 start_lr=1.0e-5,
                 end_lr=1.0e-6,
                 data_path='data',
                 vocab_file=vocab_13w,
                 vocab_size=128320,
                 load=None,
                 save=None,
                 strategy_load_ckpt_path=None,
                 finetune=False,
                 ):
        self.model = model
        self.model_parallel_size = model_parallel_size
        self.batch_size = batch_size
        self.train_iters = train_iters
        self.lr_decay_iters = int(0.64 * self.train_iters)
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.load = load
        self.finetune = finetune
        if load is not None:
            self.save = self.load if save is None else save
        else:
            self.save = save
        self.data_path = data_path
        self.vocab_file = vocab_file
        self.vocab_size = vocab_size
        self.strategy_load_ckpt_path = strategy_load_ckpt_path
        if self.finetune:
            assert self.load is not None, "> Please set your pretrained [model.ckpt] path!"
            # assert self.strategy_load_ckpt_path is not None, "> Please set your pretrained model [strategy.ckpt] path!"



    def _cover_modelarts_training_args(self, oneCardInference=False):
        global MODEL_CONFIG, DEFAULT_CONFIG
        if oneCardInference:
            tmp_config = MODEL_CONFIG[self.model]
            tmp_config['model_parallel_size'] = 1
            assert self.model_parallel_size == 1, "> mp=1 when Your using OneCardInference!"
        else:
            tmp_config = MODEL_CONFIG[self.model]
            if self.model_parallel_size == 1:
                self.model_parallel_size = tmp_config['model_parallel_size']
        default_config = copy.deepcopy({**tmp_config, **DEFAULT_CONFIG})
        _vars = vars(self)
        default_config['load'] = self.load
        default_config['save'] = self.save
        default_config['finetune'] = self.finetune
        default_config['data_path'] = self.data_path
        default_config['vocab_file'] = self.vocab_file
        default_config['vocab_size'] = self.vocab_size
        default_config['lr_decay_iters'] = self.lr_decay_iters
        # _vars.pop('checkpoint_path')
        for k, v in _vars.items():
            default_config[k] = v
        return default_config
