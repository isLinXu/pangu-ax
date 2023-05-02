#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Date: 2022/7/20
# @Author: 2022 PCL
import os
import sys
from loguru import logger

from pcl_pangu.context import check_context
from pcl_pangu.model.launcher_torch import launch
from pcl_pangu.model.evolution.config_evolution import DISTRUBUTED_CONFIG, model_config_gpu


def train(config):
    print('----------------------------- training config -----------------------------')
    print("> Base Model: [evolution]")
    print("> Model Size: [{}]".format(config.model))
    print("> data_path: {}".format(config.data_path))
    print("> global batch_size: {}".format(config.batch_size))
    print("> save to path: {}".format(config.save))
    print('------------------------------ end of config -----------------------------')

    if check_context()=='pytorch':
        script_args = config._get_training_script_args()
        py_script = '/panguAlpha_pytorch/pretrain_evolution.py'
        run_pt(script_args, py_script)

    else:
        print("ERROR: wrong backend.")
        return 1

def fine_tune(config):
    print('--------------------------- finetune config -----------------------------')
    print("> Base Model: [evolution]")
    print("> Model Size: [{}]".format(config.model))
    print("> data_path: {}".format(config.data_path))
    print("> global batch_size: {}".format(config.batch_size))
    print("> save to path: {}".format(config.save))
    print('---------------------------- end of config -------------------------------')

    if check_context()=='pytorch':
        script_args = config._get_training_script_args()
        py_script = '/panguAlpha_pytorch/pretrain_evolution.py'
        run_pt(script_args, py_script)

    else:
        print("ERROR: wrong backend.")
        return 1

def inference(config,top_k=1,top_p=0.9,input=None,input_file=None,output_file=None,
              generate_max_tokens=128,oneCardInference=True):

    backend_context = check_context()
    global output_samples, raw_text
    backend_context = check_context()
    result_output = None
    assert generate_max_tokens > 0 and generate_max_tokens<=800, "> generate_max_tokens always in (0, 800]"
    print('--------------------------- inference config --------------------------')
    print("> Base Model: [evolution]")
    print("> Model Size: [{}]".format(config.model))
    print("> global batch_size: {}".format(config.batch_size))
    print("> generate_max_tokens length: {}".format(generate_max_tokens))
    print('---------------------------- end of config ----------------------------')

    if backend_context=='pytorch':
        from .config_evolution import DEFAULT_CONFIG
        DEFAULT_CONFIG['finetune'] = True
        config.batch_size = 1
        script_args = config._get_training_script_args(oneCardInference=oneCardInference)
        py_script = '/panguAlpha_pytorch/tools/generate_samples_epangu.py'
        script_args.append('--top-k={}'.format(top_k))
        script_args.append('--top-p={}'.format(top_p))
        if input is not None:
            script_args.append('--sample-input={}'.format(input.encode('utf-8').hex()))
        if input_file is not None:
            script_args.append('--sample-input-file={}'.format(input_file))
        if output_file is not None:
            script_args.append('--sample-output-file={}'.format(output_file))
        if generate_max_tokens is not None:
            script_args.append('--generate_max_tokens={}'.format(generate_max_tokens))

        script_args.append('--isEvolution=True')
        run_pt(script_args, py_script)

    elif 'onnx-' in backend_context:
        from pcl_pangu.onnx_inference.infer import onnx_generate
        from pcl_pangu.tokenizer.tokenization_jieba import get_tokenizer

        num_threads = 2
        past_path = None
        model_path = None
        for filename in os.listdir(config.load):
            if filename.endswith('.npy'):
                past_path = os.path.join(config.load, filename)
            if filename.endswith('.onnx'):
                model_path = os.path.join(config.load, filename)

        tokenizer = get_tokenizer()
        if input is not None:
            raw_text = input
            output_samples = onnx_generate(raw_text,model_path,tokenizer,past_path,
                                           topk=top_k,top_p=top_p,threads=num_threads,
                                           max_len=generate_max_tokens, backend=backend_context)

            print('Input is:', raw_text)
            print('Output is:', output_samples[len(raw_text):], flush=True)
            print()

        if input_file is not None:
            raw_texts = open(input_file, 'r').read().split('\n\n')
            write_output = print
            if output_file is not None:
                output_file = open(output_file, 'w')
                write_output = lambda x: output_file.write(x + '\n')
            for raw_text in raw_texts:
                output_samples = onnx_generate(raw_text,model_path,tokenizer,past_path,
                                               topk=top_k,top_p=top_p,threads=num_threads,
                                               max_len=generate_max_tokens, backend=backend_context)
                write_output('Input is: ' + raw_text)
                write_output('Output is: ' + output_samples[len(raw_text):])
                write_output()

    else:
        print("ERROR: wrong backend.")
        # return 1
    if output_samples[len(raw_text):] is not None:
        result_output = output_samples[len(raw_text):]
    return result_output


def run_pt(script_args, py_script, **kwargs):
    current_dir = os.path.dirname(os.path.dirname(__file__))
    sys.path.append(current_dir + '/panguAlpha_pytorch')

    py_script = current_dir + py_script
    logger.info("> Running {} with args: {}".format(py_script, script_args))

    launch(training_script=py_script,
           training_script_args=script_args,
           **DISTRUBUTED_CONFIG,
           **kwargs)
    return 0


if __name__ == '__main__':
    config = model_config_gpu()
    inference(config)
