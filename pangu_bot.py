import gradio as gr
import random
import time

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    def respond(message, chat_history):
        bot_message = random.choice(["How are you?", "I love you", "I'm very hungry"])
        chat_history.append((message, bot_message))
        time.sleep(1)
        return "", chat_history


    from pcl_pangu.context import set_context
    from pcl_pangu.model import alpha, evolution, mPangu
    def alpha_inference(message,chat_history):
        set_context(backend='onnx-cpu')
        bot_message = random.choice(["this is pangu chat test!"])
        chat_history.append((message, bot_message))
        config = alpha.model_config_onnx(model='2B6', load='/Users/gatilin/Pan/ckpts/onnx_int8_pangu_alpha_2b6/')

        # config = alpha.model_config_onnx(model='2B6'/'13B',load='onnx/mode/path')

        return alpha.inference(config, input=message,
                               generate_max_tokens=128, output_file=None), chat_history


    # msg.submit(respond, [msg, chatbot], [msg, chatbot])

    msg.submit(alpha_inference, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch()