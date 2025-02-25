"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
from hawk.common.registry import registry
from decord import VideoReader
import decord
import numpy as np
from hawk.processors import transforms_video
from hawk.processors.base_processor import BaseProcessor
from hawk.processors.randaugment import VideoRandomAugment
from hawk.processors import functional_video as F
from omegaconf import OmegaConf
from torchvision import transforms
import random as rnd
import cv2  # Import OpenCV

MAX_INT = registry.get("MAX_INT")
decord.bridge.set_bridge("torch")

mag_threshold = 0.2

def compute_optical_flow(frames,frame_list):
    # 函数用于计算帧序列的光流
    optical_flows = []
    numpy_frame = frames.asnumpy()
    # prev_frame = cv2.cvtColor(numpy_frame[0], cv2.COLOR_RGB2GRAY)
    frame_list[0] = 1
    for i in frame_list:
        prev_frame = cv2.cvtColor(numpy_frame[i-1], cv2.COLOR_RGB2GRAY)
        current_frame = cv2.cvtColor(numpy_frame[i], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_frame, current_frame, None, 0.5, 3, 10, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        norm_mag = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX).astype(np.uint8)
        mask = (mag > mag_threshold).astype(np.uint8)
        mask = np.stack((mask, mask, mask), axis=-1)
        attention_frame = numpy_frame[i] * mask
         
        # flow_rgb = flow_to_color(flow)
        optical_flows.append(attention_frame)
        # prev_frame = current_frame

    # 将光流的特征维度
    optical_flows = np.stack(optical_flows, axis=0)

    return optical_flows

def flow_to_color(flow):
    # 将光流转换为可视化的颜色图像
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return flow_vis

def load_video_motion(video_path, n_frms=MAX_INT, height=-1, width=-1, sampling="uniform", return_msg = False):
    decord.bridge.set_bridge('native')
    vr = VideoReader(uri=video_path, height=height, width=width)

    vlen = len(vr)
    start, end = 0, vlen

    n_frms = min(n_frms, vlen)

    if sampling == "uniform":
        indices = np.arange(start, end, vlen / n_frms).astype(int).tolist()
    elif sampling == "headtail":
        indices_h = sorted(rnd.sample(range(vlen // 2), n_frms // 2))
        indices_t = sorted(rnd.sample(range(vlen // 2, vlen), n_frms // 2))
        indices = indices_h + indices_t
    else:
        raise NotImplementedError

    # get_batch -> T, H, W, C
    # temp_frms = vr.get_batch(indices)
    frames = vr.get_batch(np.arange(len(vr)))

    # print(type(frames))
    temp_frms = compute_optical_flow(frames,indices)
    
    decord.bridge.set_bridge("torch")
    
    tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
    frms = tensor_frms.permute(3, 0, 1, 2).float()  # (C, T, H, W)

    if not return_msg:
        return frms

    fps = float(vr.get_avg_fps())
    sec = ", ".join([str(round(f / fps, 1)) for f in indices])
    # " " should be added in the start and end
    msg = f"The video contains {len(indices)} frames sampled at {sec} seconds. "
    return frms, msg


def load_video(video_path, n_frms=MAX_INT, height=-1, width=-1, sampling="uniform", return_msg = False):
    decord.bridge.set_bridge("torch")
    vr = VideoReader(uri=video_path, height=height, width=width)

    vlen = len(vr)
    start, end = 0, vlen

    n_frms = min(n_frms, vlen)

    if sampling == "uniform":
        indices = np.arange(start, end, vlen / n_frms).astype(int).tolist()
    elif sampling == "headtail":
        indices_h = sorted(rnd.sample(range(vlen // 2), n_frms // 2))
        indices_t = sorted(rnd.sample(range(vlen // 2, vlen), n_frms // 2))
        indices = indices_h + indices_t
    else:
        raise NotImplementedError

    # get_batch -> T, H, W, C
    temp_frms = vr.get_batch(indices)
    # print(type(temp_frms))
    tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
    frms = tensor_frms.permute(3, 0, 1, 2).float()  # (C, T, H, W)

    if not return_msg:
        return frms

    fps = float(vr.get_avg_fps())
    sec = ", ".join([str(round(f / fps, 1)) for f in indices])
    # " " should be added in the start and end
    msg = f"The video contains {len(indices)} frames sampled at {sec} seconds. "
    return frms, msg


class AlproVideoBaseProcessor(BaseProcessor):
    def __init__(self, mean=None, std=None, n_frms=MAX_INT):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        self.normalize = transforms_video.NormalizeVideo(mean, std)

        self.n_frms = n_frms


class ToUint8(object):
    def __init__(self):
        pass

    def __call__(self, tensor):
        return tensor.to(torch.uint8)

    def __repr__(self):
        return self.__class__.__name__


class ToTHWC(object):
    """
    Args:
        clip (torch.tensor, dtype=torch.uint8): Size is (C, T, H, W)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (T, H, W, C)
    """

    def __init__(self):
        pass

    def __call__(self, tensor):
        return tensor.permute(1, 2, 3, 0)

    def __repr__(self):
        return self.__class__.__name__


class ResizeVideo(object):
    def __init__(self, target_size, interpolation_mode="bilinear"):
        self.target_size = target_size
        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: central cropping of video clip. Size is
            (C, T, crop_size, crop_size)
        """
        return F.resize(clip, self.target_size, self.interpolation_mode)

    def __repr__(self):
        return self.__class__.__name__ + "(resize_size={0})".format(self.target_size)


@registry.register_processor("alpro_video_train")
class AlproVideoTrainProcessor(AlproVideoBaseProcessor):
    def __init__(
        self,
        image_size=384,
        mean=None,
        std=None,
        min_scale=0.5,
        max_scale=1.0,
        n_frms=MAX_INT,
    ):
        super().__init__(mean=mean, std=std, n_frms=n_frms)

        self.image_size = image_size

        self.transform = transforms.Compose(
            [
                # Video size is (C, T, H, W)
                transforms_video.RandomResizedCropVideo(
                    image_size,
                    scale=(min_scale, max_scale),
                    interpolation_mode="bicubic",
                ),
                ToTHWC(),  # C, T, H, W -> T, H, W, C
                ToUint8(),
                transforms_video.ToTensorVideo(),  # T, H, W, C -> C, T, H, W
                # self.normalize,
            ]
        )

    def __call__(self, vpath):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: video clip after transforms. Size is (C, T, size, size).
        """
        clip = load_video(
            video_path=vpath,
            n_frms=self.n_frms,
            height=self.image_size,
            width=self.image_size,
            sampling="headtail",
        )
        clip_motion = load_video_motion(
            video_path=vpath,
            n_frms=self.n_frms,
            height=self.image_size,
            width=self.image_size,
            sampling="headtail",
        )

        return self.transform(clip), self.transform(clip_motion)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 256)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        min_scale = cfg.get("min_scale", 0.5)
        max_scale = cfg.get("max_scale", 1.0)

        n_frms = cfg.get("n_frms", MAX_INT)

        return cls(
            image_size=image_size,
            mean=mean,
            std=std,
            min_scale=min_scale,
            max_scale=max_scale,
            n_frms=n_frms,
        )


@registry.register_processor("alpro_video_eval")
class AlproVideoEvalProcessor(AlproVideoBaseProcessor):
    def __init__(self, image_size=256, mean=None, std=None, n_frms=MAX_INT):
        super().__init__(mean=mean, std=std, n_frms=n_frms)

        self.image_size = image_size

        # Input video size is (C, T, H, W)
        self.transform = transforms.Compose(
            [
                # frames will be resized during decord loading.
                ToUint8(),  # C, T, H, W
                ToTHWC(),  # T, H, W, C
                transforms_video.ToTensorVideo(),  # C, T, H, W
                # self.normalize,  # C, T, H, W
            ]
        )

    def __call__(self, vpath):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: video clip after transforms. Size is (C, T, size, size).
        """
        clip = load_video(
            video_path=vpath,
            n_frms=self.n_frms,
            height=self.image_size,
            width=self.image_size,
        )

        clip_motion = load_video_motion(
            video_path=vpath,
            n_frms=self.n_frms,
            height=self.image_size,
            width=self.image_size,
            sampling="headtail",
        )

        return self.transform(clip), self.transform(clip_motion)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 256)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        n_frms = cfg.get("n_frms", MAX_INT)

        return cls(image_size=image_size, mean=mean, std=std, n_frms=n_frms)
