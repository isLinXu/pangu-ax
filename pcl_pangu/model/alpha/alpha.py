#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Date: 2022/7/20
# @Author: 2022 PCL
import os
import sys
from loguru import logger
from ..launcher_torch import launch

from pcl_pangu.context import check_context
from .config_alpha import DISTRUBUTED_CONFIG, model_config_gpu, model_config_npu


def train(config):
    print('---------------------------- train config ----------------------------')
    print("> Base Model: [alpha]")
    print("> Model Size: [{}]".format(config.model))
    print("> data_path: {}".format(config.data_path))
    print("> global batch_size: {}".format(config.batch_size))
    print("> save to path: {}".format(config.save))
    print('--------------------------- end of config ----------------------------')

    if check_context()=='pytorch':
        assert isinstance(config, model_config_gpu)
        script_args = config._get_training_script_args()
        py_script = '/panguAlpha_pytorch/pretrain_gpt2.py'
        run_pt(script_args, py_script)
    elif check_context()=='mindspore':
        assert isinstance(config, model_config_npu)
        config_dict = config._cover_modelarts_training_args()
        run_ms_train(config_dict)



def fine_tune(config):
    print('-------------------------- finetune config ---------------------------')
    print("> Base Model: [alpha]")
    print("> Model Size: [{}]".format(config.model))
    print("> data_path: {}".format(config.data_path))
    print("> global batch_size: {}".format(config.batch_size))
    print("> save to path: {}".format(config.save))
    print('--------------------------- end of config ----------------------------')

    if check_context()=='pytorch':
        from .config_alpha import DEFAULT_CONFIG
        DEFAULT_CONFIG['finetune'] = True
        script_args = config._get_training_script_args()
        py_script = '/panguAlpha_pytorch/pretrain_gpt2.py'
        run_pt(script_args, py_script)
    elif check_context()=='mindspore':
        assert isinstance(config, model_config_npu)
        config_dict = config._cover_modelarts_training_args()
        run_ms_train(config_dict)
        run_ms_finetune_merge_OpenI(config_dict)


def inference(config,top_k=1,top_p=0.9,input=None,input_file=None,
              generate_max_tokens=128, output_file=None,oneCardInference=True):

    global output_samples, raw_text
    backend_context = check_context()
    result_output = ''
    assert generate_max_tokens > 0 and generate_max_tokens <= 800, "> generate_max_tokens always in (0, 800]"
    print('--------------------------- inference config --------------------------')
    print("> Base Model: [alpha]")
    print("> Model Size: [{}]".format(config.model))
    print("> global batch_size: {}".format(config.batch_size))
    print("> generate_max_tokens length: {}".format(generate_max_tokens))
    print('---------------------------- end of config ----------------------------')

    if backend_context == 'pytorch':
        from .config_alpha import DEFAULT_CONFIG
        DEFAULT_CONFIG['finetune'] = True
        config.batch_size = 1
        script_args = config._get_training_script_args(oneCardInference=oneCardInference)
        py_script = '/panguAlpha_pytorch/tools/generate_samples_Pangu.py'
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
        run_pt(script_args, py_script)

    elif backend_context == 'mindspore':
        assert isinstance(config, model_config_npu)
        config_dict = config._cover_modelarts_training_args(oneCardInference=oneCardInference)
        config_dict['top_k'] = top_k
        config_dict['top_p'] = top_p
        config_dict['input'] = input
        config_dict['input_file'] = input_file
        config_dict['output_file'] = output_file
        config_dict['generate_max_tokens'] = generate_max_tokens
        run_ms_inference(config_dict)

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
            output_samples = onnx_generate(raw_text, model_path, tokenizer, past_path,
                                           topk=top_k, top_p=top_p, threads=num_threads,
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
                output_samples = onnx_generate(raw_text, model_path, tokenizer, past_path,
                                               topk=top_k, top_p=top_p, threads=num_threads,
                                               max_len=generate_max_tokens, backend=backend_context)
                write_output('Input is: ' + raw_text)
                write_output('Output is: ' + output_samples[len(raw_text):])
                write_output()
    if output_samples[len(raw_text):] is not None:
        result_output = output_samples[len(raw_text):]
    return result_output
def run_pt(script_args, py_script, **kwargs):
    current_dir = os.path.dirname(os.path.dirname(__file__))
    sys.path.append(current_dir+'/panguAlpha_pytorch')
    py_script = current_dir + py_script
    logger.info("> Running {} with args: {}".format(py_script, script_args))
    launch(training_script=py_script,
           training_script_args=script_args,
           **DISTRUBUTED_CONFIG,
           **kwargs)


def run_ms_train(config_dict):
    current_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
    sys.path.append(current_dir + '/panguAlpha_mindspore')
    from train_alpha_ms13 import opt, setup_args, main, alpha_add_args
    new_opt = setup_args(opt, config_dict)
    new_opt = alpha_add_args(new_opt)
    main(new_opt)

def run_ms_inference(config_dict):
    current_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
    sys.path.append(current_dir + '/panguAlpha_mindspore')
    from inference_alpha_ms13 import opt, setup_args, main
    new_opt = setup_args(opt, config_dict)
    main(new_opt)

def run_ms_finetune_merge_OpenI(config_dict):
    current_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
    sys.path.append(current_dir + '/panguAlpha_mindspore')
    from train_alpha_ms13 import opt, setup_args, main, alpha_add_args, find_local_ckpt_maxStepNumber
    import mindspore as ms
    import mindspore.communication.management as D
    import moxing as mox
    import time
    D.init()
    device_num = D.get_group_size()
    rank_id = D.get_rank()
    args_opt = setup_args(opt, config_dict)
    args_opt = alpha_add_args(args_opt)

    # do merge
    sample_save_dir_rank = os.path.join(args_opt.save_checkpoint_path, f"rank_0")
    save_dir_strategy = os.path.join(args_opt.save_checkpoint_path, "strategy")
    strategy = ms.build_searched_strategy("/cache/strategy.ckpt")
    mp = args_opt.op_level_model_parallel_num
    # load mp slice modelWeight

    if rank_id % 8 == 0:
        maxStepNumber = find_local_ckpt_maxStepNumber(sample_save_dir_rank)
        sliced_parameters = []
        for i in range(mp):
            save_dir_rank = os.path.join(args_opt.save_checkpoint_path, f"rank_{i}")
            print(save_dir_rank)
            os.system("ls {}".format(save_dir_rank))
            save_ckptfile_name = args_opt.ckpt_name_prefix + '_' + str(i)

            model_prefix = os.path.join(save_dir_rank, "{}-{}_2.ckpt".format(save_ckptfile_name, maxStepNumber))
            print(model_prefix)
            param_dict = ms.load_checkpoint(model_prefix)
            adam_names = [item for item in param_dict.keys() if 'adam' in item]
            for item in adam_names:
                param_dict.pop(item)
            param_dict.pop("scale_sense")
            param_dict.pop("global_step")
            param_dict.pop("current_iterator_step")
            param_dict.pop("last_overflow_iterator_step")
            param_dict.pop("epoch_num")
            param_dict.pop("step_num")

            sliced_parameters.append(param_dict)

        merged_parameter = {}
        for key in sliced_parameters[0].keys():
            this_merged_list = [sliced_parameters[0].get(key), sliced_parameters[1].get(key)]
            this_merged_parameter = ms.merge_sliced_parameter(this_merged_list, strategy)
            merged_parameter[key] = this_merged_parameter

        # ops_cast = ops.Cast()

        param_list_fp16 = []
        for (key, value) in merged_parameter.items():
            if key == 'embedding_table':
                key = 'backbone.word_embedding.embedding_table'
            elif key == 'backbone.embedding.embedding.position_embedding.embedding_table':
                key = 'backbone.position_embedding.embedding_table'
            elif key == 'backbone.top_query_embedding_table':
                key = 'backbone.top_query_embedding.embedding_table'
            each_param_fp16 = {}
            each_param_fp16["name"] = key
            if isinstance(value.data, ms.Tensor):
                param_data_fp16 = ms.Tensor(value.data, ms.float16)
                # param_data_fp16 = ops_cast(value.data, ms.float16)
            else:
                param_data_fp16 = ms.Tensor(value.data, ms.float16)
                # param_data_fp16 = ops_cast(ms.Tensor(value.data), ms.float16)

            each_param_fp16["data"] = param_data_fp16
            param_list_fp16.append(each_param_fp16)

        ms.save_checkpoint(param_list_fp16, "/{}/alpha-2B6-merged-{}_2-fp16.ckpt".format(args_opt.save_checkpoint_path, maxStepNumber))
        try:
            mox.file.copy("/{}/alpha-2B6-merged-{}_2-fp16.ckpt".format(args_opt.save_checkpoint_path, maxStepNumber),
                          os.path.join(args_opt.train_url, "alpha-2B6-merged-fp16.ckpt"))
        except:
            pass
        f = open("/cache/merge.txt", 'w')
        f.close()
    while not os.path.exists("/cache/merge.txt"):
        time.sleep(1)

    print('>>> merge mp={} model-slices to: {} >>>\n'.format(args_opt.op_level_model_parallel_num, args_opt.train_url))


if __name__ == '__main__':
    config = model_config_gpu()
    run_pt(config)
