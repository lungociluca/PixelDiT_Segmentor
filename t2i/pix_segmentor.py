import random
import os
import torch
from PIL import Image
import numpy as np
import argparse
from typing import Optional, List
from dataclasses import dataclass, field
import pyrallis
from argparse import Namespace
import json

import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy.io as sio
import matplotlib.pyplot as plt

import config as local_config

from diffusion.model.builder import build_model, get_tokenizer_and_text_encoder
from diffusion.model.utils import get_weight_dtype, prepare_prompt_ar
from diffusion.utils.config import PixDiTConfig, model_init_config
from diffusion.utils.logger import get_root_logger

from diffusion import DPMS


@dataclass
class PixelDiTInference(PixDiTConfig):
    config: Optional[str] = "configs/PixelDiT_1024px_pixel_diffusion_stage3.yaml"
    model_path: Optional[str] = ".."
    work_dir: Optional[str] = None
    version: str = "sigma"
    txt_file: str = "asset/samples/samples_mini.txt"
    json_file: Optional[str] = None
    sample_nums: int = 1
    bs: int = 1
    cfg_scale: float = 3.5
    sampling_algo: str = "flow_dpm-solver"
    seed: int = 0
    dataset: str = "custom"
    step: int = -1
    add_label: str = ""
    tar_and_del: bool = False
    exist_time_prefix: str = ""
    gpu_id: int = 0
    custom_image_size: Optional[int] = None
    custom_height: Optional[int] = None
    custom_width: Optional[int] = None
    start_index: int = 0
    end_index: int = 30_000
    interval_guidance: List[float] = field(default_factory=lambda: [0, 1])
    ablation_selections: Optional[List[float]] = None
    ablation_key: Optional[str] = None
    if_save_dirname: bool = False
    negative_prompt: str = ""  # optional negative prompt applied at inference

def set_env(seed=0, latent_size=256):
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    for _ in range(30):
        torch.randn(1, 4, latent_size, latent_size)

def get_model():
    args = Namespace(config='configs/PixelDiT_1024px_pixel_diffusion_stage3.yaml')
    config = args = pyrallis.parse(config_class=PixelDiTInference, config_path=args.config)

    from tools.download import resolve_checkpoint
    args.model_path = resolve_checkpoint(args.model_path or "pixeldit_t2i_v1.pth")

    args.image_size = local_config.image_size
    if args.custom_image_size:
        args.image_size = args.custom_image_size
        print(f"custom_image_size: {args.image_size}")

    set_env(args.seed, args.image_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = get_root_logger()

    # only support fixed latent size currently
    latent_size = args.image_size
    max_sequence_length = config.text_encoder.model_max_length
    flow_shift = config.scheduler.flow_shift
    guidance_type = "classifier-free"
    assert (
        isinstance(args.interval_guidance, list)
        and len(args.interval_guidance) == 2
        and args.interval_guidance[0] <= args.interval_guidance[1]
    )
    args.interval_guidance = [max(0, args.interval_guidance[0]), min(1, args.interval_guidance[1])]
    default_sample_steps = 50
    sample_steps = args.step if args.step != -1 else default_sample_steps

    weight_dtype = get_weight_dtype(config.model.mixed_precision)
    logger.info(f"Inference with {weight_dtype}, guidance_type: {guidance_type}, flow_shift: {flow_shift}")

    tokenizer, text_encoder = get_tokenizer_and_text_encoder(name=config.text_encoder.text_encoder_name, device=device)

    null_caption_token = tokenizer(
        args.negative_prompt if hasattr(args, "negative_prompt") and len(args.negative_prompt) > 0 else "",
        max_length=max_sequence_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(device)
    null_caption_embs = text_encoder(null_caption_token.input_ids, null_caption_token.attention_mask)[0]

    # model setting
    model_kwargs = model_init_config(config, latent_size=latent_size)
    model = build_model(
        config.model.model, use_fp32_attention=config.model.get("fp32_attention", False), **model_kwargs
    ).to(device)
    logger.info(
        f"{model.__class__.__name__}:{config.model.model}, Model Parameters: {sum(p.numel() for p in model.parameters()):,}"
    )
    state_dict = torch.load("pixeldit_t2i_v1.pth", map_location=lambda storage, loc: storage)
    if "pos_embed" in state_dict["state_dict"]:
        del state_dict["state_dict"]["pos_embed"]

    missing, unexpected = model.load_state_dict(state_dict["state_dict"], strict=False)
    return model.eval().to(torch.bfloat16), tokenizer, text_encoder

class Pix_Segmentor(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.model, self.tokenizer, self.text_encoder = get_model()
        with open("prompts/prompts.json") as f:
            self.prompts_dict = json.load(f)
        with open(local_config.ds_config_path) as f:
            self.labels = json.load(f)


    @staticmethod
    def img_id_from_path(x):
        return x[0]["file_name"].split("/")[-1].replace(".jpg", "")

    @staticmethod
    def resize_maps(maps, new_shape):
        return F.interpolate(maps[:,None,:,:], new_shape, mode="bilinear", align_corners=False).squeeze(1)

    # get GT for output shapes and input prompts
    def get_gt(self, x):
        gt_file_path = x[0]['file_name'].replace(local_config.current_ds_paths["img_dir"], local_config.current_ds_paths["gt_dir"]) \
            .replace(".jpg", local_config.current_ds_paths["extention"])
        mask = Image.open(gt_file_path)
        mask_tensor = torch.from_numpy(np.array(mask)).unsqueeze(0)
        return mask_tensor

    def embed_condition(self, prompts):
        caption_token = self.tokenizer(
            prompts, max_length=506, padding="max_length", truncation=True, return_tensors="pt"
        ).to(local_config.device)
        select_index = [0] + list(range(-300 + 1, 0))
        caption_embs = self.text_encoder(caption_token.input_ids, caption_token.attention_mask)[0][:, None][
            :, :, select_index
        ]
        emb_masks = caption_token.attention_mask[:, select_index]
        return caption_embs, emb_masks

    def run_diffusion_model(self, image, prompts, segment_data):
        emb, emb_mask = self.embed_condition(prompts)
        model_kwargs = dict(data_info={"img_hw": image.shape[-2], "aspect_ratio": 1}, mask=emb_mask)

        dpm_solver = DPMS(
            self.model.forward_with_dpmsolver,
            condition=emb,
            uncondition=None,
            model_type="flow",
            model_kwargs=model_kwargs,
            schedule="FLOW",
            cfg_scale=3.5
        )

        # del self.tokenizer
        # del self.text_encoder

        image = torch.nn.functional.interpolate(
            image[None,:,:,:],
            size=(512, 512),
            mode="bilinear",
            align_corners=False
        )
        samples = dpm_solver.segment(
            image.repeat(len(prompts),1,1,1),
            segment_data
        )

        os.umask(0o000)
        for i, sample in enumerate(samples):
            save_path = os.path.join("trash", "file_9.jpg")
            from torchvision.utils import save_image
            save_image(sample, save_path, nrow=1, normalize=True, value_range=(-1, 1))

    @torch.no_grad()
    def forward_no_grad(self, x):
        # image_tensor = x[0]["image"].float().to(local_config.device) / 255.
        # image_tensor = (image_tensor * 2) - 1
        # TODO: figure out how to use the providex tensor, get rid of reading from disk
        import PIL.Image
        image_tensor = PIL.Image.open(f"{x[0]['file_name']}").convert("RGB")
        image_tensor = torch.from_numpy(np.array(image_tensor)).to(torch.bfloat16).to(local_config.device).permute(2, 0, 1) / 255
        image_tensor = (image_tensor * 2) - 1

        gt = self.get_gt(x)
        h, w = gt.shape[-2:]
        file_id = Pix_Segmentor.img_id_from_path(x)
        prompts = [local_config.prompt_format.format(target=current_target) for current_target in self.prompts_dict[file_id]]

        segment_data = {}
        self.run_diffusion_model(image_tensor, prompts, segment_data)
        assert segment_data["mask"][local_config.target_layer].shape[0] == len(prompts)
        prediction = segment_data["mask"][local_config.target_layer]
        print('prediciton', prediction.shape)

        prediction = Pix_Segmentor.resize_maps(prediction, (h,w))
        predictions_all = torch.zeros((len(self.labels)+1), h, w)

        cam_dict = {}
        min_max_norm = lambda x: (x - x.min()) / (x.max() - x.min())
        for i, current_label in enumerate(self.prompts_dict[file_id]):
            label_idx = self.labels.index(current_label) + 1
            predictions_all[label_idx-1] += prediction[i]

            current_pred = min_max_norm(prediction[i].to(torch.float32))
            cam_dict[str(label_idx-1)] = (current_pred * 255).cpu().numpy()

        save_path = os.path.join("sio_maps", "images", f'{file_id}.mat')
        sio.savemat(save_path, cam_dict, do_compression=True)
        
        return [{"sem_seg": predictions_all}]
    
    def forward(self, x):
        return self.forward_no_grad(x)