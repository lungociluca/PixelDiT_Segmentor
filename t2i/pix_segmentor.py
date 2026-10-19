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
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy.io as sio
import matplotlib.pyplot as plt

import config as local_config

from diffusion.model.builder import build_model, get_tokenizer_and_text_encoder
from diffusion.model.utils import get_weight_dtype, prepare_prompt_ar
from diffusion.utils.config import PixDiTConfig, model_init_config
from diffusion.utils.logger import get_root_logger

from diffusion import DPMS
from diffusion.model.builder import MODELS
from pixdit_core.pixeldit_t2i import PixDiT_T2I

# @MODELS.register_module()
# class PixDiTTrainer(nn.Module):
#     """
#     Pixel-space PixDiT trainer/wrapper for text-to-image.

#     Expects:
#     - x: [B, 3, H, W] in [-1, 1]
#     - timestep: [B]
#     - y: [B, 1, L, C] text embeddings (Gemma/Qwen), will be squeezed to [B, L, C]
#     Returns dict with key "x": predicted image in pixel space.
#     """

#     def __init__(
#         self,
#         input_size=32,
#         in_channels=3,
#         image_size=512,
#         class_dropout_prob: float = 0.0,
#         pred_sigma: bool = False,
#         learn_sigma: bool = False,
#         config=None,
#         caption_channels: int = 2304,
#         model_max_length: int = 300,
#         extra=None,
#         **kwargs,
#     ):
#         super().__init__()

#         extra = extra or {}

#         patch_size = int(extra.get("patch_size", 32))
#         num_groups = int(extra.get("num_groups", 24))
#         hidden_size = int(extra.get("hidden_size", 1920))
#         pixel_hidden_size = int(extra.get("pixel_hidden_size", extra.get("hidden_size_x", 32)))
#         pixel_attn_hidden_size = int(extra.get("pixel_attn_hidden_size", hidden_size))
#         pixel_num_groups = int(extra.get("pixel_num_groups", num_groups))
#         total_depth = int(extra.get("depth", extra.get("num_blocks", 18)))
#         patch_depth = int(extra.get("patch_depth", extra.get("num_cond_blocks", total_depth)))
#         pixel_depth = int(extra.get("pixel_depth", max(total_depth - patch_depth, 1)))
#         num_text_blocks = int(extra.get("num_text_blocks", 4))
#         txt_embed_dim = int(extra.get("txt_embed_dim", caption_channels))
#         txt_max_length = int(extra.get("txt_max_length", model_max_length))
#         use_text_rope = bool(extra.get("use_text_rope", True))
#         text_rope_theta = float(extra.get("text_rope_theta", 10000.0))
#         repa_encoder_index = int(extra.get("repa_encoder_index", -1))
#         use_pixel_abs_pos = bool(extra.get("use_pixel_abs_pos", True))
#         pit_adaln_post_modulation = bool(extra.get("pit_adaln_post_modulation", False))

#         self.core = PixDiT_T2I(
#             in_channels=in_channels,
#             num_groups=num_groups,
#             hidden_size=hidden_size,
#             pixel_hidden_size=pixel_hidden_size,
#             pixel_attn_hidden_size=pixel_attn_hidden_size,
#             pixel_num_groups=pixel_num_groups,
#             patch_depth=patch_depth,
#             pixel_depth=pixel_depth,
#             num_text_blocks=num_text_blocks,
#             patch_size=patch_size,
#             txt_embed_dim=txt_embed_dim,
#             txt_max_length=txt_max_length,
#             use_text_rope=use_text_rope,
#             text_rope_theta=text_rope_theta,
#             repa_encoder_index=repa_encoder_index,
#             use_pixel_abs_pos=use_pixel_abs_pos,
#             pit_adaln_post_modulation=pit_adaln_post_modulation,
#         )

#         self.image_size = int(image_size)
#         self.patch_size = patch_size
#         self.pred_sigma = bool(pred_sigma)
#         self.config = config
#         self._txt_embed_dim = int(txt_embed_dim)
#         if int(caption_channels) != self._txt_embed_dim:
#             raise ValueError(f"caption_channels {caption_channels} != txt_embed_dim {self._txt_embed_dim}")

#         projector_dim = 2048
#         self._repa_projector = nn.Sequential(
#             nn.Linear(self.core.hidden_size, projector_dim),
#             nn.SiLU(),
#             nn.Linear(projector_dim, projector_dim),
#             nn.SiLU(),
#             nn.Linear(projector_dim, 768),
#         )

#     def forward(self, x, timestep, y, mask=None, data_info=None, repa_tokens=None, segment_data=None, **kwargs):
#         x = x.to(self.dtype)
#         timestep = timestep.to(self.dtype)
#         if y.dim() == 4:
#             y_proc = y.squeeze(1)
#         elif y.dim() == 3:
#             y_proc = y
#         else:
#             raise ValueError("PixDiTTrainer expects y of shape [B,1,L,C] or [B,L,C]")
#         y_proc = y_proc.to(self.dtype)

#         if y_proc.shape[-1] != self._txt_embed_dim:
#             raise RuntimeError(
#                 f"PixDiTTrainer: text embedding dim {y_proc.shape[-1]} != expected {self._txt_embed_dim}. "
#                 f"Please set config.text_encoder.caption_channels to {self._txt_embed_dim} or set extra.txt_embed_dim to {y_proc.shape[-1]}."
#             )

#         if hasattr(self.core, "last_repa_tokens"):
#             self.core.last_repa_tokens = None

#         out = self.core(x, timestep, y_proc, s=None, mask=None, segment_data=segment_data)
#         repa_loss = None
#         if repa_tokens is not None:
#             repa_tokens = repa_tokens.to(self.dtype)
            
#         if repa_tokens is not None and getattr(self.core, "last_repa_tokens", None) is not None and self.core.last_repa_tokens is not None:
#             proj_tokens = self._repa_projector(self.core.last_repa_tokens)  # [B, L, 768]
#             proj_tokens = F.normalize(proj_tokens, dim=-1)
#             B, Td, C = repa_tokens.shape
#             Bu, Tu, Cu = proj_tokens.shape
#             h_u = int(Tu ** 0.5)
#             h_d = int(Td ** 0.5)
#             if h_u * h_u == Tu and h_d * h_d == Td:
#                 if Td > Tu:
#                     dino_2d = repa_tokens.permute(0, 2, 1).reshape(B, C, h_d, h_d)
#                     dino_resized = F.interpolate(dino_2d, size=(h_u, h_u), mode="bilinear", align_corners=False)
#                     dino_resized = dino_resized.flatten(2).permute(0, 2, 1)
#                     dino_resized = F.normalize(dino_resized, dim=-1)
#                     repa_loss = -((proj_tokens * dino_resized).sum(dim=-1)).mean()
#                 elif Td < Tu:
#                     usit_2d = proj_tokens.permute(0, 2, 1).reshape(Bu, Cu, h_u, h_u)
#                     usit_resized = F.interpolate(usit_2d, size=(h_d, h_d), mode="bilinear", align_corners=False)
#                     usit_resized = usit_resized.flatten(2).permute(0, 2, 1)
#                     usit_resized = F.normalize(usit_resized, dim=-1)
#                     repa_tokens = F.normalize(repa_tokens, dim=-1)
#                     repa_loss = -((usit_resized * repa_tokens).sum(dim=-1)).mean()
#                 else:
#                     repa_tokens = F.normalize(repa_tokens, dim=-1)
#                     repa_loss = -((proj_tokens * repa_tokens).sum(dim=-1)).mean()
#         return {"x": out, "repa_loss": repa_loss}

#     def forward_with_dpmsolver(self, x, timestep, y, mask=None, segment_data=None, **kwargs):
#         out = self.forward(x, timestep, y, mask, segment_data=segment_data, **kwargs)
#         if isinstance(out, dict):
#             return out["x"]
#         return out

#     @property
#     def dtype(self):
#         return next(self.parameters()).dtype


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
    # torch.set_grad_enabled(False)
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
    for param in text_encoder.parameters():
        param.requires_grad = False

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
    )
    for name, param in model.named_parameters():
        param.requires_grad = ("learnable_token" in name)
        
    model = model.to(device)
    logger.info(
        f"{model.__class__.__name__}:{config.model.model}, Model Parameters: {sum(p.numel() for p in model.parameters()):,}"
    )
    state_dict = torch.load("pixeldit_t2i_v1.pth", map_location=lambda storage, loc: storage)
    if "pos_embed" in state_dict["state_dict"]:
        del state_dict["state_dict"]["pos_embed"]

    missing, unexpected = model.load_state_dict(state_dict["state_dict"], strict=False)
    return model.eval().to(torch.bfloat16), tokenizer, text_encoder.eval()

class Pix_Segmentor(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.model, self.tokenizer, self.text_encoder = get_model()
        with open("prompts/prompts.json") as f:
            self.prompts_dict = json.load(f)
        with open(local_config.ds_config_path) as f:
            self.labels = json.load(f)
            print("LABELS", self.labels[:10], len(self.labels))
        self.emb, self.emb_mask, self.tokens_count = self.precompute_prompts()
        
        del self.tokenizer
        del self.text_encoder

    def precompute_prompts(self):
        precomputed_embs = {}
        prompts = [local_config.prompt_format.format(target=current_target) for current_target in self.labels]
        return self.embed_condition(prompts)

    @staticmethod
    def img_id_from_path(x):
        return x[0]["file_name"].split("/")[-1].replace(".jpg", "")

    @staticmethod
    def resize_maps(maps, new_shape):
        return F.interpolate(maps[:,None,:,:], new_shape, mode="bilinear", align_corners=False).squeeze(1)

    @staticmethod
    def resize_input(images, new_shape):
        if local_config.crop_size is None:
            return torch.nn.functional.interpolate(
                images[None,:,:,:],
                size=(512, 512),
                mode="bilinear",
                align_corners=False
            )
        else:
            H, W = images.shape[-2:]
            crop_h = (H // 16) * 16
            crop_w = (W // 16) * 16
            top = (H - crop_h) // 2
            left = (W - crop_w) // 2
            return images[None, :, top:top + crop_h, left:left + crop_w]

    @staticmethod
    def pad_prediction(prediction, new_shape):
        height, width = new_shape
        target_ratio = width / height
        h, w = prediction.shape[-2:]

        current_ratio = w / h

        if current_ratio < target_ratio:
            # Need to increase width
            new_w = round(h * target_ratio)
            pad_w = new_w - w

            padding = (
                pad_w // 2,
                pad_w - pad_w // 2,
                0,
                0,
            )

        elif current_ratio > target_ratio:
            # Need to increase height
            new_h = round(w / target_ratio)
            pad_h = new_h - h

            padding = (
                0,
                0,
                pad_h // 2,
                pad_h - pad_h // 2,
            )
        else:
            # Already has the correct aspect ratio
            return prediction
        return torch.nn.functional.pad(prediction, padding, mode="constant", value=0)

    def add_extra_labels(self, gt_labels):
        extra_labels_count = local_config.extra_labels_count
        all_extra_labels = [x for x in self.labels if x not in gt_labels]
        samples_extra_labels = random.sample(all_extra_labels, extra_labels_count)
        return gt_labels + samples_extra_labels
    
    @staticmethod
    def select_labels(current_labels):
        try:
            # will break if sample size < label size
            return random.sample(current_labels, local_config.select_labels_count)
        except ValueError:
            return current_labels 

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
        caption_embs_list = [self.text_encoder(caption_token.input_ids[[i]], caption_token.attention_mask[[i]])[0][:, None][
            :, :, select_index
        ] for i in range(caption_token.input_ids.shape[0])]
        caption_embs = torch.cat(caption_embs_list, dim=0).cpu()
        tokens_count = caption_token.attention_mask.sum(dim=-1)
        emb_masks = caption_token.attention_mask[:, select_index]
        return caption_embs, emb_masks, tokens_count
    
    def run_diffusion_model(self, image, prompts, target_labels, segment_data):
        emb = torch.stack([self.emb[i] for i in target_labels], dim=0)
        emb_mask = torch.stack([self.emb_mask[i] for i in target_labels], dim=0)
        tokens_count = [self.tokens_count[i] for i in target_labels]
        # emb, emb_mask, tokens_count = self.embed_condition(prompts)
        model_kwargs = dict(data_info={"img_hw": image.shape[-2], "aspect_ratio": 1}, mask=emb_mask)
        segment_data["tokens_count"] = tokens_count

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

        image = Pix_Segmentor.resize_input(image, (local_config.image_size,local_config.image_size))
        samples = dpm_solver.segment(
            image.repeat(len(target_labels),1,1,1),
            segment_data
        )

        # os.umask(0o000)
        # for i, sample in enumerate(samples):
        #     save_path = os.path.join("trash", "file_9.jpg")
        #     from torchvision.utils import save_image
        #     save_image(sample, save_path, nrow=1, normalize=True, value_range=(-1, 1))

    @torch.no_grad()
    def forward_no_grad(self, x):
        # image_tensor = x[0]["image"].float().to(local_config.device) / 255.
        # image_tensor = (image_tensor * 2) - 1
        # TODO: figure out how to use the providex tensor, get rid of reading from disk
        import PIL.Image
        image_tensor = PIL.Image.open(f"{x[0]['file_name']}").convert("RGB")
        image_tensor = torch.from_numpy(np.array(image_tensor)).to(torch.bfloat16).to(local_config.device).permute(2, 0, 1) / 255
        image_tensor = (image_tensor * 2) - 1
        original_shape = image_tensor.shape[-2:]

        gt = self.get_gt(x)
        h, w = gt.shape[-2:]
        file_id = Pix_Segmentor.img_id_from_path(x)
        target_labels = self.prompts_dict[file_id]
        label_indexes = [self.labels.index(x) for x in target_labels]
        if local_config.select_labels_count:
            label_indexes = self.select_labels(label_indexes)
        if local_config.run_on_extra_labels:
            target_labels = self.add_extra_labels(target_labels)
            label_indexes = [self.labels.index(x) for x in target_labels]

        # TODO: fix for the inputs of ADE20k with empty target lists
        assert len(label_indexes) > 0
        prompts = [local_config.prompt_format.format(target=current_target) for current_target in target_labels]
        segment_data = {"labels": target_labels, "img_id": file_id, "label_ids": label_indexes}
        if True:
            segment_data["gt"] = gt
            # segment_data["label_ids"] = [self.labels.index(t) for t in target_labels]

        self.run_diffusion_model(image_tensor, prompts, label_indexes, segment_data)
        assert segment_data["mask"][local_config.target_layer].shape[0] == min(local_config.select_labels_count, len(prompts))
        prediction = segment_data["mask"][local_config.target_layer]
        
        # pad image if crops were used
        if local_config.crop_size is not None:
            prediction = Pix_Segmentor.pad_prediction(prediction, original_shape)
        prediction = Pix_Segmentor.resize_maps(prediction, (h,w))
        predictions_all = torch.zeros((len(self.labels)+1), h, w).to(prediction.device)

        if local_config.use_learned_tokens:
            cam_dict = {}
            min_max_norm = lambda x: (x - x.min()) / (x.max() - x.min())
            for i, current_label_id in enumerate(segment_data["label_ids"]):
                current_label = self.labels[current_label_id]
                label_idx = self.labels.index(current_label) + 1
                predictions_all[label_idx-1] += prediction[i]
                if not local_config.use_learned_tokens:
                    current_pred = min_max_norm(prediction[i].to(torch.float32))
                else:
                    current_pred = prediction[i].to(torch.float32)
                cam_dict[str(label_idx-1)] = (current_pred * 255).cpu().numpy()

            save_path = os.path.join("sio_maps", "images", f'{file_id}.mat')
            sio.savemat(save_path, cam_dict, do_compression=True)
        
        return [{"sem_seg": predictions_all, "learnable_token_loss": segment_data["loss_final"]}]
    
    def forward(self, x):
        return self.forward_no_grad(x)