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

prompt_format = 'Given a user prompt, generate an "Enhanced prompt" that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:\n- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.\n- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.\nHere are examples of how to transform or refine prompts:\n- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.\n- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.\nPlease generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:\nUser Prompt: {target}'
device = "cuda"
eval_dataset = EvalDataset.VOC12
current_ds_paths = dataset_paths[eval_dataset]
ds_config_path = current_ds_paths["json"]
image_size = 512

idx_token_of_interest = 0
eval_samples_limit = 1
timestep = 0.002