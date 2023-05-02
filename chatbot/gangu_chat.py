import random

from pcl_pangu.context import set_context
from pcl_pangu.model import alpha, evolution, mPangu


# def pangu_inference_chat(message, chat_history,
#                          model='2B6', load='../ckpts/pretrained/onnx_int8_pangu_alpha_2b6',
#                          backend='onnx-cpu', gmax_tokens=800, output=None):
def pangu_inference_chat(message, chat_history,
                         model='2B6', load='../ckpts/pretrained/onnx_int8_pangu_alpha_2b6',
                         backend='onnx-cpu', gmax_tokens=800, output=None):
    '''
    pangu_inference_chat
    :param message:
    :param chat_history:
    :param model:
    :param load:
    :param backend:
    :param gmax_tokens:
    :param output:
    :return:
    '''
    set_context(backend=backend)
    bot_message = random.choice(["this is pangu chat test!"])

    chat_history.append((message, bot_message))
    if backend == 'onnx-cpu':
        config = alpha.model_config_onnx(model=model, load=load)

        return alpha.inference(config, input=message,
                               generate_max_tokens=gmax_tokens, output_file=output), chat_history
    elif backend == 'mindspore':
        config = alpha.model_config_npu(model=model, load=load)

        return alpha.inference(config, input=message,
                               generate_max_tokens=gmax_tokens, output_file=output), chat_history
    else:
        config = alpha.model_config_gpu(model=model, load=load)

        return alpha.inference(config, input=message,
                               generate_max_tokens=gmax_tokens, output_file=output), chat_history


if __name__ == '__main__':
    chat_history = []
    while True:
        message = input(">> You:")
        if message == "quit":
            break
        output, chat_history = pangu_inference_chat(message, chat_history)
        print(">> Pangu:", output)