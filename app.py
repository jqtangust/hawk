"""
Run the following command to start the demo:
    
python app.py \
    --cfg-path ./configs/eval_configs/eval.yaml \
    --model_type llama_v2 \
    --gpu-id 1
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

from hawk.common.config import Config
from hawk.common.dist_utils import get_rank
from hawk.common.registry import registry
from hawk.conversation.conversation_video import Chat, Conversation, default_conversation, SeparatorStyle,conv_llava_llama_2
import decord
decord.bridge.set_bridge('torch')

#%%
# imports modules for registration
from hawk.datasets.builders import *
from hawk.models import *
from hawk.processors import *
from hawk.runners import *
from hawk.tasks import *
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=False, default='/remote-home/share/jiaqitang/hawk/configs/eval_configs/eval.yaml', help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=1, help="specify the gpu to load the model.")
    parser.add_argument("--model_type", type=str, default='llama_v2', help="The type of LLM")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
model.eval()
vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')

# ========================================
#             Gradio Setting
# ========================================

def gradio_reset(chat_state, img_list):
    if chat_state is not None:
        chat_state.messages = []
    if img_list is not None:
        img_list = []
    return None, gr.update(value=None, interactive=True), gr.update(interactive=False),gr.update(value="Upload & Start Chat", interactive=True), chat_state, img_list

def upload_imgorvideo(gr_video, text_input, chat_state, chatbot):
    # if args.model_type == 'vicuna':
    #     chat_state = default_conversation.copy()
    # else:
    chat_state = conv_llava_llama_2.copy()
    if gr_video is None:
        return None, None, None, gr.update(interactive=True), chat_state, None
    # elif gr_img is not None and gr_video is None:
    #     print(gr_img)
    #     chatbot = chatbot + [((gr_img,), None)]
    #     chat_state.system =  "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."
    #     img_list = []
    #     llm_message = chat.upload_img(gr_img, chat_state, img_list)
    #     return gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=True, placeholder='Type and press Enter'), gr.update(value="Start Chatting", interactive=False), chat_state, img_list,chatbot
    elif gr_video is not None:
        print(gr_video)
        chatbot = chatbot + [((gr_video,), None)]
        chat_state.system =  "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."
        img_list = []
        # llm_message = chat.upload_video_without_audio(gr_video, chat_state, img_list)
        return gr.update(interactive=False), gr.update(interactive=True, placeholder='Type and press Enter'), gr.update(value="Start Chatting", interactive=False), chat_state, img_list, chatbot
    # else:
    #     # img_list = []
    #     return gr.update(interactive=False), gr.update(interactive=False, placeholder='Currently, only one input is supported'), gr.update(value="Currently, only one input is supported", interactive=False), chat_state, None,chatbot

def gradio_ask(user_message, chatbot, chat_state):
    if len(user_message) == 0:
        return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, chat_state
    chat.ask(user_message, chat_state)
    chatbot = chatbot + [[user_message, None]]
    return '', chatbot, chat_state


def gradio_answer(gr_video, chatbot, chat_state, img_list, num_beams, temperature):
    llm_message = chat.upload_video_without_audio(gr_video, chat_state, img_list)
    llm_message = chat.answer(conv=chat_state,
                              img_list=img_list,
                              num_beams=num_beams,
                              temperature=temperature,
                              max_new_tokens=300,
                              max_length=2000)[0]
    chatbot[-1][1] = llm_message
    print(chat_state.get_prompt())
    print(chat_state)
    return chatbot, chat_state, img_list.to("cpu")

title = """
<div align="center">
    <h1>Hawk: Learning to Understand Open-World Video Anomalies</h1>
</div>

<h5 align="center"> "Have eyes like a Hawk!" </h5> 

<div style="display: flex; justify-content: center; gap: 0.25rem;">
    <a href='https://github.com/jqtangust/hawk'>
        <img src='https://img.shields.io/badge/Github-Code-success' alt="GitHub Code">
    </a>
    <a href='https://huggingface.co/spaces/Jiaqi-hkust/hawk'>
        <img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue' alt="Hugging Face Spaces">
    </a>
    <a href='https://huggingface.co/Jiaqi-hkust/hawk'>
        <img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue' alt="Hugging Face Model">
    </a>
    <a href='https://arxiv.org/pdf/2405.16886'>
        <img src='https://img.shields.io/badge/Paper-PDF-red' alt="Download Paper">
    </a>
</div>

"""

cite_markdown = ("""
## Citation
The following is a BibTeX reference:
```
@inproceedings{atang2024hawk,
  title = {Hawk: Learning to Understand Open-World Video Anomalies},
  author = {Tang, Jiaqi and Lu, Hao and Wu, Ruizheng and Xu, Xiaogang and Ma, Ke and Fang, Cheng and Guo, Bin and Lu, Jiangbo and Chen, Qifeng and Chen, Ying-Cong},
  year = {2024},
  booktitle = {Neural Information Processing Systems (NeurIPS)}
}
""")

# case_note_upload = ("""
# ### We provide some examples at the bottom of the page. Simply click on them to try them out directly.
# """)

#TODO show examples below

with gr.Blocks() as demo:
    gr.Markdown(title)

    with gr.Row():
        with gr.Column(scale=0.5):
            video = gr.Video()
            # image = gr.Image(type="filepath")
            # gr.Markdown(case_note_upload)

            upload_button = gr.Button(value="Upload & Start Chat", interactive=True, variant="primary")
            clear = gr.Button("Restart")
            
            num_beams = gr.Slider(
                minimum=1,
                maximum=10,
                value=1,
                step=1,
                interactive=True,
                label="beam search numbers)",
            )
            
            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.1,
                interactive=True,
                label="Temperature",
            )
            # audio = gr.Checkbox(interactive=True, value=False, label="Audio")
        with gr.Column():
            chat_state = gr.State()
            img_list = gr.State()
            chatbot = gr.Chatbot(label='Hawk')
            text_input = gr.Textbox(label='User', placeholder='Upload your video first and start to chat.', interactive=False)
            

    with gr.Column():
        gr.Examples(examples=[
            [f"figs/examples/explosion2.mp4", "What happened in this video? "],
            [f"figs/examples/car.mp4", "What is the anomaly for the car in this video? "],
        ], inputs=[video, text_input])
        
    gr.Markdown(cite_markdown)
    upload_button.click(upload_imgorvideo, [video, text_input, chat_state, chatbot], [video, text_input, upload_button, chat_state, img_list, chatbot])
    
    start_time = time.time()
    text_input.submit(gradio_ask, [text_input, chatbot, chat_state], [text_input, chatbot, chat_state]).then(
        gradio_answer, [video, chatbot, chat_state, img_list, num_beams, temperature], [chatbot, chat_state, img_list]
    )
    end_time = time.time()
    print('Time:', end_time - start_time)
    
    clear.click(gradio_reset, [chat_state, img_list], [chatbot, video, text_input, upload_button, chat_state, img_list])
    
demo.launch(share=False)
