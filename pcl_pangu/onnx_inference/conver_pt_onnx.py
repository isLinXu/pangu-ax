import numpy as np
import random
from .pangu.config import PanguConfig
import os
from pcl_pangu.context import check_context

DEVICE = None

def convert_2_onnx(input_model_path, model, past, DEVICE):
    import torch
    batch_size= 1
    context_tokens = [1,2,3,4]
    past = torch.tensor(past)
    if DEVICE == 'cuda':
        dummy_input = {'input_ids': torch.tensor(context_tokens, device=DEVICE, dtype=torch.int).unsqueeze(0).repeat(batch_size, 1),
                       'past': past}
    else:
        dummy_input = (torch.tensor(context_tokens, device=DEVICE, dtype=torch.int).unsqueeze(0).repeat(batch_size, 1),
                       None, None, None, past)
    # model_script = torch.jit.script(model)
    input_names = ["input_0","past"]
    output_names = ["output_0","output_1"]
    torch.onnx.export(model, dummy_input, input_model_path,input_names=input_names,
                      opset_version=11,
                      output_names=output_names,
                      use_external_data_format=True,
                      do_constant_folding=True,   # will replace some of the ops that have all constant inputs, with pre-computed constant nodes.
                      dynamic_axes={'input_0':[0,1],'past':[0,1,4],'output_0':[0,1,2]})
    print('@@@@ onnx model saved to: ', input_model_path)

def quantize_onnx(input_model_path: str, output_model_path: str):
    from onnxruntime.quantization import quantize_dynamic, QuantType

    quantized_model = quantize_dynamic(
        input_model_path,
        output_model_path,
        weight_type=QuantType.QUInt8,
        use_external_data_format=True,
        # extra_options={
        #     'DisableShapeInference': True,
        # }
    )
    print('@@@@ onnx quantized model saved to: ', output_model_path)


def pt_2_onnx8(model_name, pt_path, model_config, onnx_ckpt_root_dir):
    backend_context = check_context()
    global DEVICE
    if backend_context == 'onnx-cpu':
        DEVICE = "cpu"
    elif backend_context == 'onnx-gpu':
        DEVICE = "cuda"
    print(f">>> using DEVICE: {DEVICE} >>>")

    import torch
    if DEVICE == 'cuda':
        from .pangu.model_onnx_gpu import Pangu
    else:
        from .pangu.model import Pangu
    from .pangu.utils import load_weight

    state_dict = torch.load(pt_path, map_location=DEVICE)

    print('@@@@ model_name: ', model_name)

    path_int8 = f'onnx_int8_{DEVICE}_{model_name}'
    path_int8 = os.path.join(onnx_ckpt_root_dir, path_int8)
    assert not os.path.exists(path_int8), f'\n@@@@ onnx_int8_model_path exists: {path_int8}' \
                                          f'\n@@@@ The quantification process did not run, please remove it and run again!'
    os.system(f'rm -rf {path_int8}')
    os.system(f'mkdir -p {path_int8}')
    config = PanguConfig(model_config)
    model = Pangu(config)
    model = load_weight(model, state_dict)
    model.to(DEVICE)
    model.eval()

    layer_past_path = f'{model_name}_layer_past.npy'
    layer_past_path = os.path.join(path_int8, layer_past_path)
    input_ids = torch.tensor([13], device=DEVICE, dtype=torch.long).unsqueeze(0).repeat(1, 1)
    if DEVICE == 'cuda':
        logits, past = model({'input_ids': input_ids})
    else:
        logits, past = model(input_ids, past=None, position_ids=None, lm_labels=None, token_type_ids=None)
    np.save(layer_past_path, past.detach().cpu().numpy())

    path_onnx = f'onnx_{model_name}'
    path_onnx = os.path.join(path_int8, path_onnx)
    onnx_model_path = f'{path_onnx}/{model_name}.onnx'
    os.system(f'rm -rf {path_onnx}')
    os.system(f'mkdir -p {path_onnx}')
    convert_2_onnx(onnx_model_path, model, past, DEVICE)

    onnx_int8_model_path = f'{path_int8}/{model_name}_int8.onnx'
    quantize_onnx(onnx_model_path, onnx_int8_model_path)

    os.system(f'rm -rf {path_onnx}')