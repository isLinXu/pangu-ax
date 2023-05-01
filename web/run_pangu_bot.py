import gradio as gr

from chatbot.gangu_chat import pangu_inference_chat

with gr.Blocks() as chat:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    msg.submit(pangu_inference_chat, [msg, chatbot], [msg, chatbot])

    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    chat.launch(share=True)
