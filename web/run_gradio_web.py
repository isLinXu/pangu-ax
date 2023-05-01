import gradio as gr


from pangu_infernce import alpha_inference, alpha_evolution_inference, alpha_mPangu_inference

def run_gradio(infer_model='alpha'):
    if infer_model == 'alpha':
        pangu_web = gr.Interface(fn=alpha_inference, inputs="text", outputs="text")
    elif infer_model == 'evolution':
        pangu_web = gr.Interface(fn=alpha_evolution_inference, inputs="text", outputs="text")
    elif infer_model == 'mPangu':
        pangu_web = gr.Interface(fn=alpha_mPangu_inference, inputs="text", outputs="text")

    pangu_web.launch(share=True)


if __name__ == '__main__':
    run_gradio()