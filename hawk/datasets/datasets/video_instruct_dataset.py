import os
from hawk.datasets.datasets.base_dataset import BaseDataset
from hawk.datasets.datasets.caption_datasets import CaptionDataset
import pandas as pd
import decord
from decord import VideoReader
import random
import torch
from torch.utils.data.dataloader import default_collate
from PIL import Image
from typing import Dict, Optional, Sequence
import transformers
import pathlib
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
import copy
from hawk.processors import transforms_video,AlproVideoTrainProcessor
from torchvision import transforms
from hawk.processors.video_processor import ToTHWC,ToUint8,load_video,load_video_motion
from hawk.conversation.conversation_video import Conversation,SeparatorStyle
import numpy as np

#提取Motion+Entity
import spacy

# 加载SpaCy英文模型
nlp = spacy.load("en_core_web_sm")

# Define the list of questions
Question = [
    "Can you describe the anomaly in the video?",
    "How would you detail the anomaly found in the video?",
    "What anomaly can you identify in the video?",
    "Could you explain the anomaly observed in the video?",
    "Can you point out the anomaly in the video?",
    "What's the anomaly depicted in the video?",
    "Could you specify the anomaly present in the video?",
    "How do you perceive the anomaly in the video?",
    "Can you highlight the anomaly within the video?",
    "What anomaly is noticeable in the video?",
    "Could you characterize the anomaly seen in the video?",
    "Can you detail the specific anomaly encountered in the video?",
    "How would you describe the particular anomaly in the video?",
    "What details can you provide about the anomaly in the video?",
    "Could you elucidate on the anomaly detected in the video?",
    "Can you illustrate the nature of the anomaly in the video?",
    "What features of the anomaly in the video can you describe?",
    "Could you outline the anomaly observed in the video?",
    "How does the anomaly in the video manifest?",
    "Can you clarify the aspects of the anomaly in the video?"
]


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     
def extract_actions_and_entities_sentence(sentence):
    doc = nlp(sentence)
    action_sentences = []

    for token in doc:
        # 检查是否为动词
        if token.pos_ == "VERB":
            subjects = ' and '.join(child.text for child in token.children if child.dep_ in ["nsubj", "nsubjpass"]) #主语
            objects = ' and '.join(child.text for child in token.children if child.dep_ in ["dobj", "pobj", "obj"]) #宾语
            
            # 构建包含动作和实体的句子
            action_sentence = f"{subjects} {token.text} {objects}".strip()
            action_sentences.append(action_sentence)

    return ', '.join(action_sentences)


DEFAULT_IMAGE_PATCH_TOKEN = '<ImageHere>'
video_conversation = Conversation(
    system="",
    roles=("Human", "Assistant"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)
llama_v2_video_conversation = Conversation(
    system=" ",
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)
IGNORE_INDEX = -100

class Video_Instruct_Dataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_root,num_video_query_token=32,tokenizer_name = '/mnt/workspace/ckpt/vicuna-13b/',data_type = 'video', model_type='vicuna'):
        """
        vis_root (string): Root directory of Llava images (e.g. webvid_eval/video/)
        ann_root (string): Root directory of video (e.g. webvid_eval/annotations/)
        split (string): val or test
        """
        super().__init__(vis_processor=vis_processor, text_processor=text_processor)

        data_path = pathlib.Path(ann_root)
        with data_path.open(encoding='utf-8') as f:
            self.annotation = json.load(f)

        self.num_video_query_token = num_video_query_token
        self.vis_root = vis_root
        self.resize_size = 224
        self.num_frm = 32
        self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name, use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        self.IMAGE_PATCH_TOKEN_ID = self.tokenizer.get_vocab()[DEFAULT_IMAGE_PATCH_TOKEN]

        self.transform = AlproVideoTrainProcessor(
            image_size=self.resize_size, n_frms = self.num_frm
        ).transform
        self.data_type = data_type
        self.model_type = model_type

    def _get_video_path(self, sample):
        rel_video_fp = sample['video']
        full_video_fp = os.path.join(self.vis_root,  rel_video_fp)
        return full_video_fp

    def __getitem__(self, index):
        num_retries = 10  # skip error videos
        for _ in range(num_retries):
            try:
                sample = self.annotation[index]

                video_path = self._get_video_path(sample)
                # print(video_path)
                conversation_list = sample['QA']
                
                #替换为GPT的回答
                conversation_answer = sample['description']
                
                #提取Language Motion
                # conversation_answer = extract_actions_and_entities_sentence(conversation_answer) 
                
                random_number = random.choice([0, 1])
                if random_number == 1:
                    conversation_list[0]["q"] = random.choice(Question)
                    conversation_list[0]["a"] = conversation_answer
                
                video, msg = load_video(
                    video_path=video_path,
                    n_frms=self.num_frm,
                    height=self.resize_size,
                    width=self.resize_size,
                    sampling ="uniform", return_msg = True
                )
                #读入动作视频
                video_motion, msg_motion = load_video_motion(
                    video_path=video_path,
                    n_frms=self.num_frm,
                    height=self.resize_size,
                    width=self.resize_size,
                    sampling ="uniform", return_msg = True
                )
                
                random_seed = random.randint(0, 2**32 - 1)
                setup_seed(random_seed)
                video = self.transform(video)
                video_motion = self.transform(video_motion)
                
                if 'cn' in self.data_type:
                    msg = ""
                # 添加视频<DEFAULT_IMAGE_PATCH_TOKEN>,以及msg到convsation list 0
                sources = preprocess_multimodal(copy.deepcopy(conversation_list), None, cur_token_len=self.num_video_query_token,msg = msg)
                new_sources = convert_source_vicuna_format(sources)
                
                if self.model_type =='vicuna':
                    data_dict = preprocess(
                        new_sources,
                        self.tokenizer)
                elif self.model_type =='llama_v2':
                    data_dict = preprocess_for_llama_v2(
                        new_sources,
                        self.tokenizer)
                else:
                    print('not support')
                    raise('not support')
                data_dict = dict(input_ids=data_dict["input_ids"][0],
                                labels=data_dict["labels"][0])
                # image exist in the data
                data_dict['image'] = video
                data_dict['image_motion'] = video_motion
            except:
                print(f"Failed to load examples with video: {video_path}. "
                            f"Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue
            break
        else:  
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")
        # "image_id" is kept to stay compatible with the COCO evaluation format
        return {
            "image": video,
            "image_motion": video_motion,
            "text_input": data_dict["input_ids"],
            "labels": data_dict["labels"],
            "type":'video',
        }

    def __len__(self):
        return len(self.annotation)

    def collater(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("text_input", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id) # 该函数用于将这些列表中的张量填充到相同的长度。这里使用了batch_first=True参数来指定批次维度的位置，以便在后续计算中更容易处理。填充值是self.tokenizer.pad_token_id，它是用于填充输入序列的特殊标记。
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX) #
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id), #input_ids.ne方法，它返回一个布尔张量，指示输入张量中哪些元素不等于指定值。
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
                
        if 'image_motion' in instances[0]:
            images_motion = [instance['image_motion'] for instance in instances]
            if all(x is not None and x.shape == images_motion[0].shape for x in images_motion):
                batch['images_motion'] = torch.stack(images_motion)
            else:
                batch['images_motion'] = images_motion

        batch['conv_type'] = 'multi'
        return batch

def convert_source_vicuna_format(sources):
    new_sources = []
    for source in sources:
        new_source = []
        for i, sentence in enumerate(source):
            role_0_msg = sentence['q']
            role_1_msg = sentence['a']
            new_source.append({
                'from':'human',
                'value': role_0_msg,
            })
            new_source.append({
                'from':'gpt',
                'value': role_1_msg,
            })
        new_sources.append(new_source)
    return new_sources

def preprocess_multimodal(
    conversation_list: Sequence[str],
    multimodal_cfg: dict,
    cur_token_len: int,
    msg=''
) -> Dict:
    # 将conversational list中
    is_multimodal = True
    # image_token_len = multimodal_cfg['image_token_len']
    image_token_len = cur_token_len * 2
    conversation_list[0]["q"] = "<Video>"+DEFAULT_IMAGE_PATCH_TOKEN * image_token_len +"</Video> " + msg + conversation_list[0]["q"]
    return [conversation_list]

def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "###"
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = video_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = video_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation
        
def _tokenize_fn(strings: Sequence[str],
                tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=512,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{video_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    conversations_tokenized = _tokenize_fn(conversations, tokenizer)
    input_ids = conversations_tokenized["input_ids"]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source],
                                      tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)

def preprocess_for_llama_v2(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    # add end signal and concatenate together
    conversations = []
    conv = copy.deepcopy(llama_v2_video_conversation.copy())
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    for source in sources:
        # <s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n
        header = f"<s>[INST] <<SYS>>\n{conv.system}\n</SYS>>\n\n"

        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2]
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=512,
            truncation=True,
        ).input_ids
    targets = copy.deepcopy(input_ids)


    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        # total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            
            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2 # 为什么减去2,speical token 的数目

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)
def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len
