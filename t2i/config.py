from enum import Enum

class EvalDataset(Enum):
    VOC12 = "voc_2012_test_sem_seg"
    ADE20 = "ade20k_sem_seg_val"
    # ADE20 = "ade20k_sem_seg_train"
    COCO = "coco_2017_test_stuff_all_sem_seg"

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
        "extention": ".png",
        "label_count": 150
    },
    EvalDataset.VOC12: {
        "json": "catseg_configs/voc20.json",
        "img_dir": "JPEGImages",
        "gt_dir": "SegmentationClassAug",
        "extention": ".png",
        "label_count": 20
    }
}

tmp = """
small or large [TARGET]; near or distant [TARGET]; visible, partially visible, occluded, or truncated [TARGET]; isolated, overlapping, or crowded [TARGET]; [TARGET] defined by its overall shape; [TARGET] defined by its distinctive parts; [TARGET] defined by its visual appearance, texture, or material; [TARGET] defined by its local boundaries and separation from surrounding regions.
"""

prompt_format = 'This should be a realistic image. A scene with a {target} somewhere. The {target} can be close or far. It can be a small {target} or a large {target}. There should be other stuff, for example: sky, vegetation, mutliple objects.'
device = "cuda"
eval_dataset = EvalDataset.ADE20
current_ds_paths = dataset_paths[eval_dataset]
ds_config_path = current_ds_paths["json"]
label_count = current_ds_paths["label_count"]
image_size = 512

epochs = 1
lr = 1e-4
idx_token_of_interest = 0
eval_samples_limit = 25000
target_layer = 0
layer_count = 2
timestep = 0.002
extra_labels_count = 2
select_labels_count = None

grad_accumulation = 128

no_leanable_tokens = 1
no_vision_learnable_tokens = 150
l2_regularization_weight = 0.
diversity_regularization_weight = 0.

shuffle = True
run_on_extra_labels = False
use_attention_bias = False
use_vision_attention_bias = False
use_vision_mlp = True
use_text_mlp = True
norm_before_dot = True

crop_size = None
compute_model_vectors = False
use_model_vectors = False

# TODO: update
save_learned_tokens = True 
use_learned_tokens = not save_learned_tokens