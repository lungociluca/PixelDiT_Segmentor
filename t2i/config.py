from enum import Enum

class EvalDataset(Enum):
    VOC12 = "voc_2012_test_sem_seg"
    ADE20 = "ade20k_sem_seg_val"

class SegmentDataKeys(Enum):
    MASK = "mask"
    

ds_config_dict = {
    EvalDataset.VOC12: "catseg_configs/voc20.json",
    EvalDataset.ADE20: "catseg_configs/ade150.json"
}

dataset_paths = {
    EvalDataset.ADE20: {
        "json": "catseg_configs/ade150.json",
        "img_dir": "images",
        "gt_dir": "annotations",
        "extention": ".png"
    },
    EvalDataset.VOC12: {
        "json": "catseg_configs/voc20.json",
        "img_dir": "JPEGImages",
        "gt_dir": "SegmentationClassAug",
        "extention": ".png"
    }
}

tmp = """
small or large [TARGET]; near or distant [TARGET]; visible, partially visible, occluded, or truncated [TARGET]; isolated, overlapping, or crowded [TARGET]; [TARGET] defined by its overall shape; [TARGET] defined by its distinctive parts; [TARGET] defined by its visual appearance, texture, or material; [TARGET] defined by its local boundaries and separation from surrounding regions.
"""

prompt_format = 'This should be a realistic image. A scene with a {target} somewhere. The {target} can be close or far. It can be a small {target} or a large {target}. There should be other stuff, for example: sky, vegetation, mutliple objects.'
device = "cuda"
eval_dataset = EvalDataset.VOC12
current_ds_paths = dataset_paths[eval_dataset]
ds_config_path = current_ds_paths["json"]
image_size = 512

epochs = 1
lr = 1e-4
idx_token_of_interest = 0
eval_samples_limit = 1500
target_layer = 0
layer_count = 3
timestep = 0.002
extra_labels_count = 2

grad_accumulation = 128

no_leanable_tokens = 128
l2_regularization_weight = 0.
diversity_regularization_weight = 0.

run_on_extra_labels = False
crop_size = True
compute_model_vectors = False
use_model_vectors = False

# TODO: update
save_learned_tokens = False
use_learned_tokens = not save_learned_tokens