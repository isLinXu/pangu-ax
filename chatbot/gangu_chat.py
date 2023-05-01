import random

from pcl_pangu.context import set_context
from pcl_pangu.model import alpha, evolution, mPangu


def pangu_inference_chat(message, chat_history,
                    model='2B6',load='ckpts/onnx_int8_pangu_alpha_2b6/',
                    backend='onnx-cpu',gmax_tokens=800, output=None):
    set_context(backend=backend)
    bot_message = random.choice(["this is pangu chat test!"])
    chat_history.append((message, bot_message))
    config = alpha.model_config_onnx(model=model, load=load)

    return alpha.inference(config, input=message,
                           generate_max_tokens=gmax_tokens, output_file=output), chat_history

