from enum import Enum

class EvalDataset(Enum):
    VOC12 = "voc_2012_test_sem_seg"
    ADE20 = "ade20k_sem_seg_val"

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

device = "cuda"
eval_dataset = EvalDataset.VOC12
current_ds_paths = dataset_paths[eval_dataset]
ds_config_path = current_ds_paths["json"]
image_size = 512

prompt_format = "{target}"
idx_token_of_interest = 0
eval_samples_limit = 1