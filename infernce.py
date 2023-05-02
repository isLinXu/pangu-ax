from pcl_pangu.context import set_context
from pcl_pangu.model import alpha, evolution, mPangu


def alpha_inference(input, input_file=None, model='2B6', load='../ckpts/pretrained/onnx_int8_pangu_alpha_2b6',
                    backend='onnx-cpu', gmax_tokens=800, output=None, top_k=2, top_p=0.9, batch_size=8):
    '''
    alpha_inference
    :param input: prompt
    :param model: model='350M'/'2B6'/'13B'
    :param load: 'onnx/mode/path'
    :return: output
    '''
    set_context(backend=backend)
    if backend == 'onnx-cpu':
        # onnx
        alpha_config = alpha.model_config_onnx(model=model, load=load)
    elif backend == 'mindspore':
        # mindspore
        alpha_config = alpha.model_config_npu(model=model, model_parallel_size=1,
                                              batch_size=batch_size, load=None, save=None)
    else:
        # pytorch
        alpha_config = alpha.model_config_gpu(model=model, model_parallel_size=1,
                                              batch_size=8, load=None, save=None)

    return alpha.inference(alpha_config, top_k=top_k, top_p=top_p, input=input,
                           input_file=input_file, generate_max_tokens=gmax_tokens, output_file=output)


def alpha_evolution_inference(input,  input_file=None, model='2B6', load='./ckpts/pretrained/onnx_int8_pangu_evolution_2b6/',
                              backend='onnx-cpu', gmax_tokens=800, output=None, top_k=1, top_p=0.9,batch_size=8):
    '''
    alpha_inference
    :param input: prompt
    :param model: model='2B6'/'13B'
    :param load: 'onnx/mode/path'
    :return: output
    '''
    set_context(backend=backend)
    if backend == 'onnx-cpu':
        # onnx
        evolution_config = evolution.model_config_onnx(model=model, load=load)
    else:
        # pytorch
        evolution_config = evolution.model_config_gpu(model=model, model_parallel_size=1,
                                              batch_size=batch_size, load=None, save=None)

    return evolution.inference(evolution_config, top_k=top_k, top_p=top_p, input=input,
                           input_file=input_file, generate_max_tokens=gmax_tokens, output_file=output)


def alpha_mPangu_inference(input, model='2B6', load='./ckpts/pretrained/onnx_int8_pangu_alpha_2b6/',
                           backend='mindspore', gmax_tokens=800, output=None):
    '''
    alpha_mPangu_inference
    :param input: prompt
    :param model: model='2B6'
    :param load: 'onnx/mode/path'
    :return: output
    '''
    set_context(backend=backend)
    if backend == 'mindspore':
        mPangu_config = mPangu.model_config_npu(model=model, load=load)
    else:
        mPangu_config = mPangu.model_config_gpu(model=model, load=load)

    return mPangu.inference(mPangu_config, input=input,
                            generate_max_tokens=gmax_tokens, output_file=output)


if __name__ == '__main__':
    output = alpha_inference("西红柿炒蛋怎么做?")
    print("alpha_inference output: ", output)

    # output = alpha_evolution_inference("西红柿炒蛋怎么做?")
    # print("alpha_evolution_inference output: ", output)
