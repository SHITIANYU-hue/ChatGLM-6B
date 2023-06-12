from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModel,GenerationConfig
import uvicorn
import json
import datetime
import torch
from peft import get_peft_model, LoraConfig, TaskType
import gradio as gr
import mdtex2html

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE

"""Override Chatbot.postprocess"""


def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text


# def predict(input, chatbot, max_length, top_p, temperature, history):
#     chatbot.append((parse_text(input), ""))
#     for response, history in model.stream_chat(tokenizer, input, history, max_length=max_length, top_p=top_p,
#                                                temperature=temperature):
#         chatbot[-1] = (parse_text(input), parse_text(response))       

#         yield chatbot, history

def predict(input, chatbot, max_length, top_p, temperature, history):
    chatbot.append((parse_text(input), ""))

    def generate_prompt(input):
        if input!=None:
            return f"""网友说：{input},用可爱的风格回复道："""
        else:
            return f"""网友说：{input},用可爱的风格回复道："""

    input_=generate_prompt(input)
    print('previous input',input)
    print('after input',input_)
    for response, history in model.stream_chat(tokenizer, input_, history, max_length=max_length, top_p=top_p,
                                               temperature=temperature):
        chatbot[-1] = (parse_text(input), parse_text(response))       
        print('chatbot',chatbot)
        yield chatbot, history
def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], []


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">可爱机器人</h1>""")

    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(
                    container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_length = gr.Slider(0, 4096, value=2048, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.7, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)

    history = gr.State([])

    submitBtn.click(predict, [user_input, chatbot, max_length, top_p, temperature, history], [chatbot, history],
                    show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)



if __name__ == '__main__':

    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    tokenizer = AutoTokenizer.from_pretrained(
        "THUDM/chatglm-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained(
        "THUDM/chatglm-6b", trust_remote_code=True).half().cuda()

    LOAD_8BIT = True

    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    TARGET_MODULES = ['query_key_value']

    peft_path = "output/checkpoint-52000/adapter_model.bin"



    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=True,
        r=8,
        lora_alpha=32, lora_dropout=0.1
    )

    model = get_peft_model(model, peft_config)
    model.load_state_dict(torch.load(peft_path), strict=False)

    model=model.eval()
    demo.queue().launch(share=True, inbrowser=True) 

    

