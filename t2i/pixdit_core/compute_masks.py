import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import einops

import config as local_config
from san_criterion import HungarianMatcher, SetCriterion


W_CLASS, W_MASK, W_DICE = 2.0, 5.0, 5.0
NUM_CLASSES = 171            # your vocabulary size, WITHOUT the +1
NUM_POINTS  = 12544        # SAN default (112*112); point-sampled, not dense

matcher = HungarianMatcher(
    cost_class=W_CLASS, cost_mask=W_MASK, cost_dice=W_DICE,
    num_points=NUM_POINTS,
)
weight_dict = {"loss_ce": W_CLASS, "loss_mask": W_MASK, "loss_dice": W_DICE}

criterion = SetCriterion(
    num_classes=NUM_CLASSES,
    matcher=matcher,
    weight_dict=weight_dict,
    eos_coef=0.1,                 # no-object weight
    losses=["labels", "masks"],
    num_points=NUM_POINTS,
    oversample_ratio=3.0,
    importance_sample_ratio=0.75,
)


def get_gt_bin(gt, label_ids, h, w):
    gt[gt == 255] = 0
    gt_binary = torch.nn.functional.one_hot(gt.to(torch.int64), num_classes=local_config.label_count+1)
    gt_binary = F.interpolate(gt_binary.permute(2,0,1).unsqueeze(0).to(torch.bfloat16), (h, w), mode="nearest")[0]
    gt_binary =  einops.rearrange(gt_binary, "b h w -> b (h w)")
    return torch.stack([gt_binary[lid] for lid in label_ids], dim=0)


# def save_intermediary_plots(merged_mask, gt_bin, gt_ids, h, w):
#     import matplotlib.pyplot as plt
#     import os
#     if os.path.isfile("print"):
#         with torch.no_grad():
#             for ii in range(gt_bin.shape[0]):
#                 pred = merged_mask[ii].detach().float().cpu().view(h,w)
#                 gt = gt_bin[ii].detach().float().cpu().view(h,w)

#                 fig, axes = plt.subplots(1, 2, figsize=(10, 5))

#                 axes[0].imshow(pred, cmap="hot")
#                 axes[0].set_title(f"pred of {gt_ids[ii]}")
#                 axes[0].axis("off")

#                 axes[1].imshow(gt, cmap="hot",)
#                 axes[1].set_title("gt_bin")
#                 axes[1].axis("off")

#                 plt.tight_layout()
#                 plt.savefig(f"trash/{len(os.listdir('trash'))}.jpg")
#                 plt.close(fig)
#         # os.remove("print")


def min_max(masks, dim):
    # reduce over the last dim, keep dims for broadcasting
    mins = masks.amin(dim=dim, keepdim=True)
    maxs = masks.amax(dim=dim, keepdim=True)
    return (masks - mins) / (maxs - mins + 1e-8)

def clasify_masks(visual_tokens, language_tokens, language_weights, language_no_object, label_ids):
    # WARNING: language_tokens are kept on CPU
    # visual_tokens -> (l, d), language_tokens -> (t,d)
    visual_tokens = visual_tokens.sum(dim=1) # sum across layers TODO: try other ways of aggregating visual features for comparison with language
    print("visual tokens", visual_tokens.shape, "language tokens", language_tokens.shape)
    # TODO: In SAN paper the comparison is done over the entire vocabulary, not just over gt classes
    language_tokens_selected = torch.stack([language_tokens[i] for i in range(171)], dim=0).to(local_config.device)
    print("language_tokens_selected", language_tokens_selected.shape)
    language_targets = language_weights * language_tokens_selected
    language_targets = language_targets.sum(dim=-2)
    language_targets = torch.cat([language_targets, language_no_object[None,:,:]], dim=0).permute(0,2,1)
    print("language_targets + no obj", language_targets.shape)
    classes = visual_tokens @ language_targets
    print("classes", classes.shape)
    return classes


def compute_masks(features, tokens, vl_features, language_tokens, learnable_token_norm_weights, language_weights, language_no_object, label_ids):
    B, L, P, D = features.shape
    my_norm = lambda x: x if not local_config.norm_before_dot else torch.nn.functional.normalize(x, dim=-1)
    # min_max = lambda x: (x - x.min()) / (x.max() - x.min())
    masks = my_norm(features) @ my_norm(tokens).permute(0, 1, 3, 2)

    # masks (b, l, p, t) * (b, l, 1, t)
    masks = learnable_token_norm_weights(masks)
    masks = masks.sum(dim=1) # TODO: Sigmoid()? Tanh?

    mask_classes = clasify_masks(vl_features, language_tokens, language_weights, language_no_object, label_ids)
    return masks, mask_classes.squeeze(-1).unsqueeze(0)

# def compute_loss(features, tokens, vl_features, language_tokens, learnable_token_norm_weights, language_weights, gt, label_ids, h, w):
#     B, L, P, D = features.shape
#     gt_bin = get_gt_bin(gt, label_ids, h, w).to(local_config.device)

#     mask_logits, mask_classes = compute_masks(features, tokens, vl_features, language_tokens, learnable_token_norm_weights, language_weights, label_ids)

#     bce = torch.nn.functional.binary_cross_entropy_with_logits(mask_logits, gt_bin, reduction="mean")
#     intersection = (merged_mask.sigmoid() * gt_bin).sum(dim=-1)
#     union = merged_mask.sigmoid().sum(dim=-1) + gt_bin.sum(dim=-1)
#     dice = 1.0 - (2.0 * intersection + 1e-7) / (union + 1e-7)
#     dice = dice.mean()
#     bce_weight = 0.5
#     return bce_weight * bce + (1.0 - bce_weight) * dice, dice.clone().detach()

def compute_loss(features, tokens, vl_features, language_tokens,
                 learnable_token_norm_weights, language_weights, language_no_object, gt, label_ids, h, w):
    B, L, P, D = features.shape
    gt_bin = get_gt_bin(gt, label_ids, h, w).to(local_config.device)

    mask_logits, mask_classes_logits = compute_masks(features, tokens, vl_features,
                                              language_tokens, learnable_token_norm_weights,
                                              language_weights, language_no_object, label_ids)

    mask_logits = mask_logits.permute(0,2,1).view(1, local_config.no_vision_learnable_tokens, h, w)
    mask_classes_logits = mask_classes_logits.permute(0,2,1)

    outputs = {
        "pred_logits": mask_classes_logits,   # [B, Q, C+1]   last col = no-object
        "pred_masks":  mask_logits,    # [B, Q, Hm, Wm]
    }
    targets = [
        {"labels": torch.LongTensor(label_ids).to(local_config.device),        # [T_b]  long, ids into the C vocabulary
         "masks":  gt_bin.view(len(label_ids), h, w)}         # [T_b, Hg, Wg]  {0,1}, padding masks removed
    ]
    losses = criterion(outputs, targets)
    return sum(losses[k] * weight_dict[k] for k in losses if k in weight_dict), losses["loss_dice"]

# W_CLASS, W_MASK, W_DICE = 2.0, 5.0, 5.0
# NUM_CLASSES = 171            # your vocabulary size, WITHOUT the +1
# NUM_POINTS  = 12544        # SAN default (112*112); point-sampled, not dense

# matcher = HungarianMatcher(
#     cost_class=W_CLASS, cost_mask=W_MASK, cost_dice=W_DICE,
#     num_points=NUM_POINTS,
# )
# weight_dict = {"loss_ce": W_CLASS, "loss_mask": W_MASK, "loss_dice": W_DICE}

# criterion = SetCriterion(
#     num_classes=NUM_CLASSES,
#     matcher=matcher,
#     weight_dict=weight_dict,
#     eos_coef=0.1,                 # no-object weight
#     losses=["labels", "masks"],
#     num_points=NUM_POINTS,
#     oversample_ratio=3.0,
#     importance_sample_ratio=0.75,
# )


def compute_loss000(gt, h, w):
    # Per image loss; gt is x[0]["sem_seg"], shape [Hg, Wg], values in [0, C-1] plus 255 = ignore
    mask_proposal_count = 40
    classes_count = 171  # COCO-Stuff

    # --- hardcoded dummy predictions ---
    mask_logits = torch.zeros((mask_proposal_count, h, w), device=local_config.device)
    mask_logits[2, h // 2:, w // 2:] += 0.8                      # query 2 fires on bottom-right

    mask_classes_logits = torch.zeros((mask_proposal_count, classes_count + 1),
                                      device=local_config.device)
    mask_classes_logits[:, 1] += 1.0                             # every query leans toward class 1

    outputs = {
        "pred_logits": mask_classes_logits[None],   # [1, Q, C+1]
        "pred_masks":  mask_logits[None],           # [1, Q, h, w]
    }

    # --- targets built from the real gt ---
    gt = gt.to(local_config.device)
    present = torch.unique(gt)
    present = present[present != 255]               # drop ignore
    present = present[present < classes_count]      # safety

    labels = present.long().to(local_config.device)                                     # [T]
    masks  = torch.stack([(gt == c) for c in present]).float().to(local_config.device)  # [T, Hg, Wg], {0,1}

    targets = [{"labels": labels, "masks": masks}]

    with open("expected_in.txt", "w") as f:
        f.write(f"pred logits: {mask_classes_logits[None].shape}\npred_masks {mask_logits[None].shape}\nlabels {labels.shape}\nmasks {masks.shape}\n")
    losses = criterion(outputs, targets)
    total = sum(losses[k] * weight_dict[k] for k in losses if k in weight_dict)
    return mask_logits, total, losses["loss_dice"]
    
def prepare_targets(targets, images):
    h_pad, w_pad = images.tensor.shape[-2:]
    new_targets = []
    for targets_per_image in targets:
        # pad gt
        gt_masks = targets_per_image.gt_masks
        padded_masks = torch.zeros(
            (gt_masks.shape[0], h_pad, w_pad),
            dtype=gt_masks.dtype,
            device=gt_masks.device,
        )
        padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
        new_targets.append(
            {
                "labels": targets_per_image.gt_classes,
                "masks": padded_masks,
            }
        )
    return new_targets