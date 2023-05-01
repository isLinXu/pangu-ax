'''
    code by TaeHwan Jung(@graykode)
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT
'''
import numpy as np
import os
from .pangu.sample import generate
import onnxruntime

SESSION = None
INT8_MODEL_PATH = None

def create_model_for_provider(model_path: str, provider: str= 'CPUExecutionProvider',threads=2):
    from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions, get_all_providers

    assert provider in get_all_providers(), f"provider {provider} not found, {get_all_providers()}"
    # # Few properties that might have an impact on performances (provided by MS)
    options = SessionOptions()
    # options.intra_op_num_threads = int(os.environ.get('NUM_THREADS', 2))
    # options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    # Load the model as a graph and prepare the CPU backend
    session = InferenceSession(model_path, options, providers=[provider])
    # session = InferenceSession(model_path, providers=[provider])
    session.disable_fallback()

    return session

def onnx_infer_func(onnx_model_path: str,threads=2, backend='onnx-cpu'):
    input_names = ["input_0","past"]
    global SESSION
    if INT8_MODEL_PATH != onnx_model_path:
        if backend == 'onnx-cpu':
            SESSION = create_model_for_provider(onnx_model_path, 'CPUExecutionProvider', threads=threads)
        elif backend == 'onnx-gpu':
            SESSION = create_model_for_provider(onnx_model_path, 'CUDAExecutionProvider', threads=threads)

    def forward(input_ids,
                position_ids,
                token_type_ids,
                lm_labels,
                past):
        return SESSION.run(None, {
            input_names[0]: input_ids.astype(np.int32),
            input_names[1]: past,
        })

    return forward

def onnx_generate(input,onnx_int8_model_path,tokenizer,past_path,
                   topk=1,top_p=0.9,threads=2,max_len=500, backend='onnx-cpu'):
    assert os.path.exists(onnx_int8_model_path), f"{onnx_int8_model_path} not found"
    assert onnx_int8_model_path.endswith('.onnx'), f"{onnx_int8_model_path} is not a onnx model"
    assert past_path.endswith('.npy'), f"{past_path} is not a npy file"

    past=np.load(past_path)
    infer_func = onnx_infer_func(onnx_int8_model_path,threads=threads, backend=backend)
    txt = generate(infer_func, input, tokenizer, past,max_len=max_len,
                   top_p=top_p, top_k=topk,temperature=1.0,)

    return txt

if __name__ == '__main__':
    from pcl_pangu.tokenizer.tokenization_jieba import get_tokenizer

    tokenizer = get_tokenizer()



    # generate_ = lambda text,max_len=100: generate(infer_func, text, tokenizer, past,max_len=max_len)
    # print(generate_('青椒肉丝的做法：\n'))
    # print(generate_('西红柿炒鸡蛋的做法：\n'))
    # print(generate_('上联：天地在我心中\n下联：'))
    # print(generate_('1+1=2;3+5=8;2+4=', max_len=2))
    # print(generate_('默写古诗：\n白日依山尽，黄河入海流。\n床前明月光，', max_len=5))
    # print(generate_('李大嘴：“各回各家，各找各妈！” \n佟掌柜：'))
    # print(generate_('中国的首都是北京\n日本的首都是东京\n美国的首都是'))
    # print(generate_('中国的四大发明有哪些？', 50))
    # print(generate_('''乔布斯曾经说过：“''', 50))
    # print(generate_('''老子曾经说过：“''', 50))
    # print(generate_('''老子曾经说过：“''', 50))


