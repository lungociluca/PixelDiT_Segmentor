import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json, einops
from typing import Tuple

from .pixeldit_c2i import PatchTokenEmbedder, PixelTokenEmbedder, PiTBlock
from .modules import (
    FinalLayer,
    FeedForward,
    RMSNorm,
    TimestepConditioner,
    apply_adaln,
    apply_rotary_emb,
    precompute_freqs_cis_2d,
)

import config as local_config

class MMDiTJointAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            block_id: int = -1

    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv_x = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_y = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.q_norm_x = RMSNorm(self.head_dim)
        self.k_norm_x = RMSNorm(self.head_dim)
        self.q_norm_y = RMSNorm(self.head_dim)
        self.k_norm_y = RMSNorm(self.head_dim)

        self.proj_x = nn.Linear(dim, dim)
        self.proj_y = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop_x = nn.Dropout(proj_drop)
        self.proj_drop_y = nn.Dropout(proj_drop)
        self.block_id = block_id

        with open(local_config.ds_config_path) as f:
            self.labels = json.load(f)

    def forward(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            pos_img: torch.Tensor,
            pos_txt: torch.Tensor = None,
            attn_mask: torch.Tensor = None,
            segment_data=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # TODO: check corectness: slicing y so that it matches the dimensions of rotary embeddings
        B, _, C = y.shape
        pos_emb_size = y.shape[1] - local_config.no_leanable_tokens
        qy_learned, ky_learned, vy_learned  = self.qkv_y(y[:,:pos_emb_size]).reshape(B, pos_emb_size, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
        qy_learned = self.q_norm_y(qy_learned)
        ky_learned = self.k_norm_y(ky_learned)
        if pos_txt is not None:
            qy_learned, ky_learned = apply_rotary_emb(qy_learned, ky_learned, freqs_cis=pos_txt)
        qy_learned = qy_learned.transpose(1, 2)
        ky_learned = ky_learned.transpose(1, 2)
        vy_learned = vy_learned.transpose(1, 2)
        out_joint_learned = F.scaled_dot_product_attention(qy_learned, ky_learned, vy_learned, dropout_p=0.0)[:,:,:local_config.no_leanable_tokens,:]
        out_y_learned = out_joint_learned.transpose(1, 2).reshape(B, local_config.no_leanable_tokens, C)
        # TODO: implement dropout
        out_y_learned = self.proj_y(out_y_learned)
        segment_data["learned"] = out_y_learned

        # remove learnable tokens after they attented with prompt tokens
        y = y[:, local_config.no_leanable_tokens:, :]

        B, Nx, C = x.shape
        By, Ny, Cy = y.shape
        assert B == By and C == Cy, "x and y must share batch and channel dims"

        qkv_x = self.qkv_x(x).reshape(B, Nx, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
        qx, kx, vx = qkv_x[0], qkv_x[1], qkv_x[2]
        qx = self.q_norm_x(qx)
        kx = self.k_norm_x(kx)

        qkv_y = self.qkv_y(y).reshape(B, Ny, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
        qy, ky, vy = qkv_y[0], qkv_y[1], qkv_y[2]
        qy = self.q_norm_y(qy)
        ky = self.k_norm_y(ky)

        my_norm = lambda x: x
        target_tokens_y = torch.stack([ky[ii, [segment_data["tokens_count"][ii].item() - 1], :, :] for ii in range(B)], dim=0)
        attn_mk = (my_norm(qx.permute(0,2,1,3)) @ target_tokens_y.permute(0,2,3,1)).mean(1).squeeze(-1)

        with torch.enable_grad():
            tmp = F.mse_loss(self.learnable_token.weight, torch.ones((local_config.no_leanable_tokens, self.dim), device=local_config.device))
            print("LOSS", tmp.item())
        segment_data["loss"] = tmp

        qx, kx = apply_rotary_emb(qx, kx, freqs_cis=pos_img)
        if pos_txt is not None:
            qy, ky = apply_rotary_emb(qy, ky, freqs_cis=pos_txt)

        qx = qx.transpose(1, 2)
        kx = kx.transpose(1, 2)
        vx = vx.transpose(1, 2)

        qy = qy.transpose(1, 2)
        ky = ky.transpose(1, 2)
        vy = vy.transpose(1, 2)

        q_joint = torch.cat([qy, qx], dim=2)
        k_joint = torch.cat([ky, kx], dim=2)
        v_joint = torch.cat([vy, vx], dim=2)

        out_joint = F.scaled_dot_product_attention(q_joint, k_joint, v_joint, dropout_p=0.0, attn_mask=attn_mask)
        out_y = out_joint[:, :, :Ny, :]
        out_x = out_joint[:, :, Ny:, :]

        out_y = out_y.transpose(1, 2).reshape(B, Ny, C)
        out_x = out_x.transpose(1, 2).reshape(B, Nx, C)

        out_x = self.proj_drop_x(self.proj_x(out_x))
        out_y = self.proj_drop_y(self.proj_y(out_y))
        return out_x, out_y, attn_mk

class MMDiTBlockT2I(nn.Module):
    def __init__(self, hidden_size, groups, mlp_ratio=4.0, adaLN_modulation_img=None, adaLN_modulation_txt=None, block_id=-1):
        super().__init__()
        self.hidden_size = hidden_size
        self.groups = groups
        self.head_dim = hidden_size // groups

        self.norm_x1 = RMSNorm(hidden_size, eps=1e-6)
        self.norm_y1 = RMSNorm(hidden_size, eps=1e-6)
        self.norm_learnable_token = RMSNorm(hidden_size, eps=1e-6)

        self.attn = MMDiTJointAttention(hidden_size, num_heads=groups, qkv_bias=False, block_id=block_id)

        self.norm_x2 = RMSNorm(hidden_size, eps=1e-6)
        self.norm_y2 = RMSNorm(hidden_size, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp_x = FeedForward(hidden_size, mlp_hidden_dim)
        self.mlp_y = FeedForward(hidden_size, mlp_hidden_dim)

        self.adaLN_modulation_img = adaLN_modulation_img if adaLN_modulation_img is not None else nn.Sequential(nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.adaLN_modulation_txt = adaLN_modulation_txt if adaLN_modulation_txt is not None else nn.Sequential(nn.Linear(hidden_size, 6 * hidden_size, bias=True))

        self.block_id = block_id
        self.learnable_token = nn.Embedding(local_config.no_leanable_tokens, hidden_size)

    # def gather_features(self, features, segment_data):
    #     B, P, D = features.shape
    #     gt = segment_data["gt"]
    #     label_ids = segment_data["label_ids"]

    #     gt[gt == 255] = 0
    #     gt_binary = torch.nn.functional.one_hot(gt[0].to(torch.int64), num_classes=21)
    #     gt_binary = F.interpolate(gt_binary.permute(2,0,1).unsqueeze(0).to(features.dtype), (32, 32), mode="nearest")[0]
    #     gt_binary = einops.rearrange(gt_binary, "b h w -> b (h w)")
        
    #     targets = []
    #     for ii in range(B):
    #         gt_class_features = torch.stack(
    #             [features[0, pid] for pid in range(P) if gt_binary[label_ids[ii], pid] == 1]
    #         )
    #         targets.append(gt_class_features.mean(dim=0))
    #     segment_data["targets"] = torch.stack(targets)
    #     segment_data["gt_binary"] = gt_binary

    @staticmethod
    def get_gt_bin(segment_data):
        gt = segment_data["gt"]
        label_ids = segment_data["label_ids"]
        h, w = segment_data["hw"]

        gt[gt == 255] = 0
        gt_binary = torch.nn.functional.one_hot(gt[0].to(torch.int64), num_classes=21)
        gt_binary = F.interpolate(gt_binary.permute(2,0,1).unsqueeze(0).to(torch.bfloat16), (h, w), mode="nearest")[0]
        gt_binary =  einops.rearrange(gt_binary, "b h w -> b (h w)")
        return torch.stack([gt_binary[lid] for lid in label_ids], dim=0)

    def compute_loss(self, features, segment_data):
        B, P, D = features.shape
        gt_bin = self.get_gt_bin(segment_data).to(local_config.device)
        # TODO
        tokens = self.norm_learnable_token(self.learnable_token.weight)
        masks = features @ tokens.expand(B, -1, -1).permute(0, 2, 1)
        merged_mask = masks.sum(dim=-1)
        merged_mask = merged_mask.sigmoid()

        intersection = (gt_bin * merged_mask).sum(dim=-1)
        union = merged_mask.sum(dim=-1) + gt_bin.sum(dim=-1) - intersection
        iou = (intersection + 1e-7) / (union + 1e-7)
        # TODO: does this add less importance to examples with 2 classes per image than 1?
        return 1.0 - iou.mean()

    def forward(self, x, y, c, pos_img, pos_txt=None, attn_mask=None, segment_data=None):
        print("tokens", [self.learnable_token.weight[ii].mean().item() for ii in range(4)])
        shift_msa_x, scale_msa_x, gate_msa_x, shift_mlp_x, scale_mlp_x, gate_mlp_x = self.adaLN_modulation_img(c).chunk(6, dim=-1)
        shift_msa_y, scale_msa_y, gate_msa_y, shift_mlp_y, scale_mlp_y, gate_mlp_y = self.adaLN_modulation_txt(c).chunk(6, dim=-1)

        # add learnable tokens
        with torch.enable_grad():

            y = torch.cat([self.learnable_token.weight.expand(y.shape[0], -1, -1), y], dim=1)
            # y = torch.cat([torch.load("learnable_token_0.pt", weights_only=False).to("cuda").to(torch.bfloat16).weight.expand(y.shape[0], -1, -1), y], dim=1)
            
            x_norm = apply_adaln(self.norm_x1(x), shift_msa_x, scale_msa_x)
            y_norm = apply_adaln(self.norm_y1(y), shift_msa_y, scale_msa_y)
            attn_x, attn_y, attn_mk = self.attn(x_norm, y_norm, pos_img, pos_txt, attn_mask, segment_data=segment_data)
            loss = self.compute_loss(x, segment_data)
            print("LOSS", loss.item())
            
        segment_data["loss"] = tmp

        # remove learnable tokens
        y = y[:, local_config.no_leanable_tokens:, :]

        x = x + gate_msa_x * attn_x
        y = y + gate_msa_y * attn_y

        x = x + gate_mlp_x * self.mlp_x(apply_adaln(self.norm_x2(x), shift_mlp_x, scale_mlp_x))
        y = y + gate_mlp_y * self.mlp_y(apply_adaln(self.norm_y2(y), shift_mlp_y, scale_mlp_y))
        return x, y, attn_mk

    def __del__(self):
        if local_config.save_learned_tokens:
            torch.save(self.learnable_token, f"learnable_token_{self.block_id}.pt")

class PixDiT_T2I(nn.Module):
    def __init__(
        self,
        in_channels=3,
        num_groups=16,
        hidden_size=1152,
        pixel_hidden_size=64,
        pixel_attn_hidden_size=None,
        pixel_num_groups=None,
        patch_depth=26,
        pixel_depth=2,
        num_text_blocks=4,
        patch_size=16,
        txt_embed_dim=4096,
        txt_max_length=1024,
        use_text_rope: bool = True,
        text_rope_theta: float = 10000.0,
        repa_encoder_index: int = -1,
        use_pixel_abs_pos: bool = True,
        pit_adaln_post_modulation: bool = False,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(in_channels)
        self.hidden_size = int(hidden_size)
        self.num_groups = int(num_groups)
        self.patch_depth = int(patch_depth)
        self.pixel_depth = int(pixel_depth)
        self.num_text_blocks = int(num_text_blocks)
        self.patch_size = int(patch_size)
        self.pixel_hidden_size = int(pixel_hidden_size)
        self.txt_embed_dim = int(txt_embed_dim)
        self.txt_max_length = int(txt_max_length)
        self.use_text_rope = bool(use_text_rope)
        self.text_rope_theta = float(text_rope_theta)
        self.repa_encoder_index = int(repa_encoder_index)
        self.use_pixel_abs_pos = bool(use_pixel_abs_pos)
        self.pit_adaln_post_modulation = bool(pit_adaln_post_modulation)
        if self.pixel_depth <= 0:
            raise ValueError("PixDiT_T2I expects pixel_depth > 0 to retain the pixel pathway")

        self.pixel_embedder = PixelTokenEmbedder(in_channels, self.pixel_hidden_size, use_pixel_abs_pos=self.use_pixel_abs_pos)
        self.s_embedder = PatchTokenEmbedder(in_channels * patch_size ** 2, hidden_size, bias=True)
        self.t_embedder = TimestepConditioner(hidden_size)
        self.y_embedder = PatchTokenEmbedder(self.txt_embed_dim, hidden_size, bias=True, norm_layer=RMSNorm)
        self.y_pos_embedding = nn.Parameter(torch.randn(1, self.txt_max_length, hidden_size))
        # self.register_buffer("learnable_token_target", torch.ones(1, 1, hidden_size), persistent=False)

        self._shared_cond_adaln = None
        self._shared_cond_adaln_img = None
        self._shared_cond_adaln_txt = None
        self.patch_blocks = nn.ModuleList([
            MMDiTBlockT2I(
                self.hidden_size,
                self.num_groups,
                adaLN_modulation_img=self._shared_cond_adaln_img,
                adaLN_modulation_txt=self._shared_cond_adaln_txt,
                block_id=i
            )
            for i in range(min(self.patch_depth, local_config.layer_count))
        ])
        self.text_refine_blocks = None
        self.pixel_attn_hidden_size = (
            int(pixel_attn_hidden_size) if pixel_attn_hidden_size is not None else self.hidden_size
        )
        self.pixel_num_groups = int(pixel_num_groups) if pixel_num_groups is not None else self.num_groups
        self.pixel_blocks = nn.ModuleList(
            [
                PiTBlock(
                    self.pixel_hidden_size,
                    self.hidden_size,
                    patch_size=self.patch_size,
                    num_heads=self.num_groups,
                    mlp_ratio=4.0,
                    attn_hidden_size=self.pixel_attn_hidden_size,
                    attn_num_heads=self.pixel_num_groups,
                    rope_fn=precompute_freqs_cis_2d,
                    adaln_post_modulation=self.pit_adaln_post_modulation,
                )
                for _ in range(self.pixel_depth)
            ]
        )

        self.final_layer = FinalLayer(self.pixel_hidden_size, self.out_channels)

        self.precompute_pos = dict()
        self.precompute_pos_txt = dict()
        self.last_repa_tokens = None

        self.initialize_weights()

    def fetch_pos(self, height, width, device):
        if (height, width) in self.precompute_pos:
            return self.precompute_pos[(height, width)].to(device)
        else:
            pos = precompute_freqs_cis_2d(self.hidden_size // self.num_groups, height, width).to(device)
            self.precompute_pos[(height, width)] = pos
            return pos

    def fetch_pos_text(self, length, device):
        if length in self.precompute_pos_txt:
            return self.precompute_pos_txt[length].to(device)
        head_dim = self.hidden_size // self.num_groups
        freqs = 1.0 / (self.text_rope_theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
        positions = torch.arange(0, length, device=device).float().unsqueeze(1)
        angles = positions * freqs.unsqueeze(0)
        freqs_cis = torch.polar(torch.ones_like(angles), angles)
        self.precompute_pos_txt[length] = freqs_cis
        return freqs_cis

    def initialize_weights(self):
        w = self.s_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.s_embedder.proj.bias, 0)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def forward(self, x, t, y, s=None, mask=None, segment_data=None):
        B, _, H, W = x.shape
        Hs = H // self.patch_size
        Ws = W // self.patch_size
        L = Hs * Ws

        pos = self.fetch_pos(Hs, Ws, x.device)
        x_patches = torch.nn.functional.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)

        t_emb = self.t_embedder(t.view(-1)).view(B, -1, self.hidden_size)

        if y.dim() != 3:
            raise ValueError("Text embedding y must be [B, L, D]")
        Ltxt = min(y.shape[1], max(1, self.txt_max_length - 1))
        y = y[:, :Ltxt, :]
        y_emb = self.y_embedder(y).view(B, Ltxt, self.hidden_size)
        y_emb = y_emb + self.y_pos_embedding[:, :Ltxt, :].to(y_emb.dtype)


        condition = torch.nn.functional.silu(t_emb)
        if segment_data is not None:
            segment_data["mask"] = {}
            segment_data["hw"] = (Hs,Ws)
        if s is None:
            s0 = self.s_embedder(x_patches)
            pos_txt = self.fetch_pos_text(Ltxt, x.device) if self.use_text_rope else None
            attn_mask_joint = None
            if mask is not None and isinstance(mask, torch.Tensor):
                m = mask
                while m.dim() > 2 and m.size(1) == 1:
                    m = m.squeeze(1)
                if m.dim() == 3 and m.size(1) == 1:
                    m = m.squeeze(1)
                if m.dim() == 2:
                    pad = (m == 0)
                    pad_img = torch.zeros((B, L), dtype=torch.bool, device=x.device)
                    attn_mask_joint = torch.cat([pad[:, :Ltxt], pad_img], dim=1).view(B, 1, 1, Ltxt + L)
            self.last_repa_tokens = None
            s = s0
            for i in range(self.patch_depth):
                s, y_emb, attn_mk = self.patch_blocks[i](s, y_emb, condition, pos, pos_txt, attn_mask_joint, segment_data=segment_data)
                if 0 < self.repa_encoder_index == (i + 1):
                    self.last_repa_tokens = s
                
                # import os
                # attn_mk_disp = attn_mk.to(torch.float32)[-1,:].view(Hs, Ws).detach().cpu()
                # plt.imshow(attn_mk_disp, cmap="viridis")
                # plt.savefig(f"trash/{i}_{len(os.listdir('trash'))}.png")
                # plt.close()

                if segment_data is not None:
                    segment_data["mask"][i] = attn_mk.view(B, Hs, Ws)
                    if i == local_config.layer_count - 1:
                        break

            s = torch.nn.functional.silu(t_emb + s)
        if not (0 < self.repa_encoder_index <= self.patch_depth):
            self.last_repa_tokens = s

        batch_size, length, _ = s.shape
        if length != L:
            if length > L:
                s = s[:, :L, :]
            else:
                pad_len = L - length
                s = torch.cat([s, s.new_zeros(B, pad_len, s.shape[2])], dim=1)
            length = L

        s_cond = s.view(B * L, self.hidden_size)
        x_pixels = self.pixel_embedder(x, img_height=H, img_width=W, patch_size=self.patch_size)
        for blk in self.pixel_blocks:
            x_pixels = blk(x_pixels, s_cond, H, W, self.patch_size, mask)

        x_pixels = self.final_layer(x_pixels)
        C_out = self.out_channels
        P2 = self.patch_size * self.patch_size
        x_pixels = x_pixels.view(B, L, P2, C_out).permute(0, 3, 2, 1).contiguous()
        x_pixels = x_pixels.view(B, C_out * P2, L)
        x_img = torch.nn.functional.fold(x_pixels, (H, W), kernel_size=self.patch_size, stride=self.patch_size)
        return x_img
