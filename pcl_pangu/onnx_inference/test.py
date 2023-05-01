'''
    code by TaeHwan Jung(@graykode)
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT
'''
import os
import sys
import torch
import random
import argparse
import numpy as np
from pangu.model import (Pangu)
from pangu.utils import load_weight
from pangu.config import PanguConfig
from pangu.sample import sample_sequence, generate
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions, get_all_providers
from tokenization_jieba import JIEBATokenizer

tokenizer = JIEBATokenizer(
    'bpe_4w_pcl/vocab.vocab',
    'bpe_4w_pcl/vocab.model')



def create_model_for_provider(model_path: str, provider: str= 'CPUExecutionProvider') -> InferenceSession:
    assert provider in get_all_providers(), f"provider {provider} not found, {get_all_providers()}"
    # Few properties that might have an impact on performances (provided by MS)
    options = SessionOptions()
    options.intra_op_num_threads = int(os.environ.get('NUM_THREADS', 2))
    print(f"@@@@ Using {options.intra_op_num_threads} threads for inference")
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    # Load the model as a graph and prepare the CPU backend
    session = InferenceSession(model_path, options, providers=[provider])
    session.disable_fallback()

    return session


def convert_2_onnx(input_model_path, model, past):
    batch_size= 1
    context_tokens = [1,2,3,4]
    past = torch.tensor(past)
    dummy_input = (torch.tensor(context_tokens, device="cpu", dtype=torch.int).unsqueeze(0).repeat(batch_size, 1),
                   None,None,None,past)
    # model_script = torch.jit.script(model)
    input_names = ["input_0","past"]
    output_names = ["output_0","output_1"]
    torch.onnx.export(model, dummy_input, input_model_path,input_names=input_names,
                      opset_version=11,
                      output_names=output_names,
                      use_external_data_format=True,
                      dynamic_axes={'input_0':[0,1],'past':[0,1,4],'output_0':[0,1,2]})
    print('@@@@ onnx model saved to: ', input_model_path)

def quantize_onnx(input_model_path: str, output_model_path: str):
    quantized_model = quantize_dynamic(
        input_model_path,
        output_model_path,
        weight_type=QuantType.QUInt8,
        use_external_data_format=True,
        extra_options={
            'DisableShapeInference': True,
        }
    )
    print('@@@@ onnx quantized model saved to: ', output_model_path)

def onnx_infer_func(onnx_model_path: str):
    input_names = ["input_0","past"]
    session = create_model_for_provider(onnx_model_path, 'CPUExecutionProvider')
    def forward(input_ids,
                position_ids,
                token_type_ids,
                lm_labels,
                past):
        return session.run(None, {
            input_names[0]: input_ids.astype(np.int32),
            input_names[1]: past,
        })
    return forward

def torch_infer_func(model):
    def forward(input_ids,
                position_ids,
                token_type_ids,
                lm_labels,
                past):
        input_ids = torch.tensor(input_ids)
        past = torch.tensor(past)
        logits, past = model(input_ids, past=past, position_ids=None, lm_labels=None, token_type_ids=None)
        logits = logits.detach().cpu().numpy()
        return logits, past
    return forward

def text_generator(model_name, model_size, onnx_ckpt_root_dir='', infer_backend='onnx',state_dict=None):
    seed = random.randint(0, 2147483647)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = "cpu"

    print('@@@@ model_name: ', model_name)

    if state_dict is not None:
        # Load Model
        config = PanguConfig(model_size=model_size)
        model = Pangu(config)
        model = load_weight(model, state_dict)
        model.to(device)
        model.eval()

    layer_past_path = f'{model_name}_layer_past.npy'
    if not os.path.exists(layer_past_path):
        input_ids = torch.tensor([13], device=device, dtype=torch.long).unsqueeze(0).repeat(1, 1)
        logits, past = model(input_ids, past=None, position_ids=None, lm_labels=None, token_type_ids=None)
        np.save(layer_past_path, past.detach().cpu().numpy())
    past = np.load(layer_past_path)

    path = f'onnx_{model_name}'
    path = os.path.join(onnx_ckpt_root_dir, path)
    onnx_model_path = f'{path}/{model_name}.onnx'
    if not os.path.exists(onnx_model_path):
        os.system(f'rm -rf {path}')
        os.system(f'mkdir -p {path}')
        convert_2_onnx(onnx_model_path, model, past)

    path = f'onnx_int8_{model_name}'
    path = os.path.join(onnx_ckpt_root_dir, path)
    onnx_int8_model_path = f'{path}/{model_name}_int8.onnx'
    if not os.path.exists(onnx_int8_model_path):
        os.system(f'rm -rf {path}')
        os.system(f'mkdir -p {path}')
        quantize_onnx(onnx_model_path, onnx_int8_model_path)

    if infer_backend == 'onnx':
        print('@@@@ onnx running on device: ', ort.get_device())
        print('@@@@ onnx inference')
        infer_func = onnx_infer_func(onnx_int8_model_path)
    elif infer_backend == 'torch':
        print('@@@@ torch inference')
        infer_func = torch_infer_func(model)

    generate_ = lambda text,max_len=100: generate(infer_func, text, tokenizer, past,max_len=max_len)

    print(generate_('青椒肉丝的做法：\n'))
    print(generate_('西红柿炒鸡蛋的做法：\n'))
    print(generate_('上联：天地在我心中\n下联：'))
    print(generate_('1+1=2;3+5=8;2+4=', max_len=2))
    print(generate_('默写古诗：\n白日依山尽，黄河入海流。\n床前明月光，', max_len=5))
    print(generate_('李大嘴：“各回各家，各找各妈！” \n佟掌柜：'))
    print(generate_('中国的首都是北京\n日本的首都是东京\n美国的首都是'))
    print(generate_('中国的四大发明有哪些？', 50))
    print(generate_('''乔布斯曾经说过：“''', 50))
    print(generate_('''老子曾经说过：“''', 50))
    print(generate_('''老子曾经说过：“''', 50))

if __name__ == '__main__':
    # model_name = 'pangu_evolution'
    # model_size = '2b6'
    # text_generator(model_name, model_size)
    # # model_path = '/home/yands/tmp/models/pangu-alpha-evolution_2.6b_fp16/iter_0055000/mp_rank_00/model_optim_rng.pt'
    # # assert os.path.exists(model_path)
    # # state_dict = torch.load(model_path, map_location='cpu')
    # # text_generator(model_name, model_size, infer_backend='torch',state_dict=state_dict)

    model_name = 'pangu_alpha_2b6'
    model_size = '2b6'
    text_generator(model_name, model_size)
    # model_path = '/raid0/yands/tmp/models/panguAlpha_2.6b_NumpyCkpt/merged/iter_0076000/mp_rank_00/model_optim_rng.pt'
    # assert os.path.exists(model_path)
    # state_dict = torch.load(model_path, map_location='cpu')
    # text_generator(model_name, model_size, infer_backend='torch',state_dict=state_dict)

    # model_name = 'pangu_alpha_13b'
    # model_size = '13b'
    # # text_generator(model_name, model_size)
    # model_path = '/raid0/yands/tmp/models/panguAlpha_13b_fp16_NumpyCkpt/merged/iter_0076000/mp_rank_00/model_optim_rng.pt'
    # assert os.path.exists(model_path)
    # state_dict = torch.load(model_path, map_location='cpu')
    # text_generator(model_name, model_size, infer_backend='torch',state_dict=state_dict)


