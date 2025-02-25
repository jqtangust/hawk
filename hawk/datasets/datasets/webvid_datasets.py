"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from hawk.datasets.datasets.base_dataset import BaseDataset
from hawk.datasets.datasets.caption_datasets import CaptionDataset
import pandas as pd
import decord
from decord import VideoReader
import random
import torch
from torch.utils.data.dataloader import default_collate
import spacy
import numpy as np

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     
# 加载SpaCy英文模型
nlp = spacy.load("en_core_web_sm")

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

class WebvidDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_root):
        """
        vis_root (string): Root directory of video (e.g. webvid_eval/video/)
        ann_root (string): Root directory of video (e.g. webvid_eval/annotations/)
        split (string): val or test
        """
        super().__init__(vis_processor=vis_processor, text_processor=text_processor)


        # 读取一个路径下所有的
        ts_df = []
        for file_name in os.listdir(ann_root):
            if file_name.endswith('.csv'):
                df = pd.read_csv(os.path.join(ann_root, file_name))
                ts_df.append(df)

        print(ts_df)
        merged_df = pd.concat(ts_df)
        
        self.annotation = merged_df
        self.vis_root = vis_root
        self.resize_size = 224
        self.num_frm = 32
        self.frm_sampling_strategy = 'headtail'

    def _get_video_path(self, sample):
        rel_video_fp = os.path.join(str(sample['page_dir']), str(sample['videoid']) + '.mp4')
        full_video_fp = os.path.join(self.vis_root,  rel_video_fp)
        return full_video_fp

    def __getitem__(self, index):
        num_retries = 10  # skip error videos
        for _ in range(num_retries):
            
            sample = self.annotation.iloc[index]
            sample_dict = sample.to_dict()
            # video_id = sample_dict['videoid']
            # fetch video
            video_path = self._get_video_path(sample_dict)

            # while not os.path.exists(video_path):
            #     index = random.randint(0, len(self.annotation) - 1)
            #     sample = self.annotation.iloc[index]
            #     sample_dict = sample.to_dict()
            #     video_path = self._get_video_path(sample_dict)

            while not os.path.exists(video_path) or (os.path.exists(video_path) and os.path.getsize(video_path) == 0):
                index = random.randint(0, len(self.annotation) - 1)
                sample = self.annotation.iloc[index]
                sample_dict = sample.to_dict()
                video_path = self._get_video_path(sample_dict)
            
            if 'name' in sample_dict.keys():
                text = sample_dict['name'].strip()
                text_motion = extract_actions_and_entities_sentence(text)
            else:
                raise NotImplementedError("Un-supported text annotation format.")
            
            # if os.path.exists(video_path):
            try:
                random_seed = random.randint(0, 2**32 - 1)
                setup_seed(random_seed)
                video, video_motion = self.vis_processor(video_path)
            except:
                print(f"for A Failed to load examples with video: {video_path}. "
                            f"Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue
            
            # text = extract_actions_and_entities_sentence(text)
            caption = self.text_processor(text)
            caption_motion = self.text_processor(text_motion)

            # print(video.size())
            if video is None or caption is None or video.size()!=torch.Size([3,self.vis_processor.n_frms,224,224]):
                print(f"for B Failed to load examples with video: {video_path}. "
                            f"Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue
            else:
                break
        else:  
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")
        # "image_id" is kept to stay compatible with the COCO evaluation format
        return {
            "image": video, #torch.Size([3, 32, 224, 224])
            "image_motion": video_motion, #torch.Size([3, 32, 224, 224])
            "text_input": caption,
            "text_input_motion": caption_motion,
            "type":'video',
        }

    def __len__(self):
        return len(self.annotation)

    # def collater(self, samples):
    #     new_result = {}
    #     new_result['image'] = default_collate( [sample["image"] for sample in samples])
    #     new_result['image_motion'] = default_collate( [sample["image_motion"] for sample in samples])
    #     new_result['text_input'] = default_collate( [sample["text_input"] for sample in samples])
    #     return new_result
        
class WebvidDatasetEvalDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):

        ann = self.annotation[index]

        vname = ann["video"]
        video_path = os.path.join(self.vis_root, vname)

        video = self.vis_processor(video_path)

        return {
            "video": video,
            "image_id": ann["image_id"],
            "instance_id": ann["instance_id"],
        }


