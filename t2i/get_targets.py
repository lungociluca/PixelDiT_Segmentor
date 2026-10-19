import argparse
import json
from pathlib import Path
from typing import Iterable, List

import numpy as np
from PIL import Image

import config as local_config

with open(local_config.ds_config_path) as f:
    text_labels = json.load(f)

def get_image_id(mask_path: Path) -> str:
    """Return the image ID, dropping '_seg' from original ADE20K mask names."""
    name = mask_path.stem
    if name.endswith("_seg"):
        name = name[:-4]
    return name


def extract_class_ids(mask_path: Path, ignore_labels: set[int]) -> List[int]:
    """Extract sorted unique class IDs from one ADE20K annotation mask."""
    with Image.open(mask_path) as image:
        mask = np.asarray(image)

    if mask.ndim == 2:
        # Standard ADEChallengeData2016 semantic masks are indexed/palette masks.
        labels = np.unique(mask)
    elif mask.ndim == 3 and mask.shape[2] >= 3:
        rgb = mask[..., :3]
        # Some semantic masks are RGB images with the same ID copied to each channel.
        if np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(
            rgb[..., 0], rgb[..., 2]
        ):
            labels = np.unique(rgb[..., 0])
        elif mask_path.stem.endswith("_seg"):
            # Original ADE20K format encodes class IDs across R and G channels.
            red = rgb[..., 0].astype(np.int32)
            green = rgb[..., 1].astype(np.int32)
            labels = np.unique((red // 10) * 256 + green)
        else:
            raise ValueError(
                f"{mask_path} is an RGB mask, but its name does not end in '_seg'. "
                "Expected a single-channel semantic mask or an original ADE20K "
                "*_seg.png annotation."
            )
    else:
        raise ValueError(
            f"Unsupported annotation shape {mask.shape} in {mask_path}; "
            "expected a 2D mask or an RGB/RGBA image."
        )

    try:
        return sorted(text_labels[int(label)-1] for label in labels if int(label) not in ignore_labels)
    except IndexError:
        print(sorted(int(label) for label in labels if int(label) not in ignore_labels))
        exit(-1)

def find_annotation_files(root: Path) -> List[Path]:
    """Find masks recursively, preferring original ADE20K *_seg.png files if present."""
    png_files = sorted(path for path in root.rglob("*.png") if path.is_file())
    original_seg_files = [path for path in png_files if path.stem.endswith("_seg")]
    if original_seg_files:
        # Avoid treating *_parts_*.png and other PNGs as whole-image masks.
        return original_seg_files
    return png_files


def build_index(
    annotations_dir: Path, ignore_labels: Iterable[int]
) -> dict[str, List[int]]:
    if not annotations_dir.exists():
        raise FileNotFoundError(f"Annotation directory does not exist: {annotations_dir}")
    if not annotations_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {annotations_dir}")

    annotation_files = find_annotation_files(annotations_dir)
    if not annotation_files:
        raise FileNotFoundError(f"No PNG annotation masks found under {annotations_dir}")

    ignored = set(ignore_labels)
    index: dict[str, List[int]] = {}
    for mask_path in annotation_files:
        image_id = get_image_id(mask_path)
        if image_id in index:
            raise ValueError(
                f"Duplicate image ID {image_id!r}. Point the script at one split/folder "
                "at a time, or adjust the image-ID generation."
            )
        index[image_id] = extract_class_ids(mask_path, ignored)

    return dict(sorted(index.items()))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a JSON mapping from ADE20K image IDs to class IDs present."
    )
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        required=True,
        help="Directory containing annotation PNGs; subdirectories are searched recursively.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ade20k_classes.json"),
        help="Output JSON file (default: ade20k_classes.json).",
    )
    parser.add_argument(
        "--ignore-labels",
        type=int,
        nargs="*",
        default=[0],
        help="Pixel label IDs to omit (default: 0, commonly used for unlabeled/ignored pixels).",
    )
    args = parser.parse_args()

    index = build_index(args.annotations_dir, args.ignore_labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(index, file, indent=2, ensure_ascii=False)
        file.write("\n")

    print(f"Indexed {len(index)} annotations -> {args.output}")


if __name__ == "__main__":
    main()
