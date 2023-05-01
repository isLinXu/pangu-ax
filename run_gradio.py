import gradio as gr

def greet(name):
    return "Hello " + name + "!"


from pcl_pangu.context import set_context
from pcl_pangu.model import alpha, evolution, mPangu

def alpha_inference(input):

    set_context(backend='onnx-cpu')

    config = alpha.model_config_onnx(model='2B6',load='/Users/gatilin/Pan/ckpts/onnx_int8_pangu_alpha_2b6/')

    # config = alpha.model_config_onnx(model='2B6'/'13B',load='onnx/mode/path')

    return alpha.inference(config,input=input,
                           generate_max_tokens=800, output_file=None)

# output = alpha_inference("四川的省会是?")
# print("output: ", output)

demo = gr.Interface(fn=alpha_inference, inputs="text", outputs="text")

demo.launch(share=True)