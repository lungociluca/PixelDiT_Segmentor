import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
from PIL import Image

with open("catseg_configs/coco.json") as f:
    classes_list = json.load(f)

def load_label_names(labels_path: Optional[Path]) -> Optional[Dict[int, str]]:
    """Load an ID->name map, or return None to emit numeric IDs.

    Accepts the official COCO-Stuff 'labels.txt' (lines of 'ID: name'),
    a JSON object {"id": "name"}, or a JSON list indexed by label ID.
    """
    if labels_path is None:
        return None
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file does not exist: {labels_path}")

    text = labels_path.read_text(encoding="utf-8")

    if labels_path.suffix.lower() == ".json":
        data = json.loads(text)
        if isinstance(data, dict):
            return {int(key): value for key, value in data.items()}
        if isinstance(data, list):
            return {index: name for index, name in enumerate(data)}
        raise ValueError(f"Unsupported JSON labels structure in {labels_path}.")

    mapping: Dict[int, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(
                f"Malformed line in {labels_path!r}: {line!r}. "
                "Expected 'ID: name' (COCO-Stuff labels.txt format)."
            )
        id_part, name = line.split(":", 1)
        mapping[int(id_part)] = name.strip()
    return mapping


def get_image_id(mask_path: Path) -> str:
    """Return the image ID. COCO-Stuff masks share the stem of the matching .jpg."""
    return mask_path.stem


def extract_class_ids(
    mask_path: Path,
    ignore_labels: set[int],
    id_to_name: Optional[Dict[int, str]],
) -> List:
    """Extract sorted unique classes from one COCO-Stuff stuffthingmaps PNG."""
    with Image.open(mask_path) as image:
        mask = np.asarray(image)

    if mask.ndim == 3:
        # stuffthingmaps are single-channel; if a PNG decodes to RGB, the ID is
        # copied across channels, so any channel carries the label.
        mask = mask[..., 0]
    elif mask.ndim != 2:
        raise ValueError(
            f"Unsupported annotation shape {mask.shape} in {mask_path}; "
            "expected a 2D indexed mask."
        )
    try:
        labels = [classes_list[int(value)-1] for value in np.unique(mask) if int(value) not in ignore_labels]
    except IndexError:
        print("index", np.unique(mask), len(classes_list), "ignore", ignore_labels)
        print("ignore", [int(value) not in ignore_labels for value in np.unique(mask)])
        exit(-1)
    if id_to_name is None:
        return sorted(labels)
    try:
        return sorted(id_to_name[label] for label in labels)
    except KeyError as error:
        raise KeyError(
            f"Label {error} in {mask_path} is not in the labels file. "
            "Check that --labels matches this map's ID scheme."
        ) from error


def find_annotation_files(root: Path) -> List[Path]:
    """Find all mask PNGs recursively (e.g. one per image under train2017/ val2017/)."""
    return sorted(path for path in root.rglob("*.png") if path.is_file())


def build_index(
    annotations_dir: Path,
    ignore_labels: Iterable[int],
    id_to_name: Optional[Dict[int, str]],
) -> Dict[str, List]:
    if not annotations_dir.exists():
        raise FileNotFoundError(f"Annotation directory does not exist: {annotations_dir}")
    if not annotations_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {annotations_dir}")

    annotation_files = find_annotation_files(annotations_dir)
    if not annotation_files:
        raise FileNotFoundError(f"No PNG annotation masks found under {annotations_dir}")

    ignored = set(ignore_labels)
    index: Dict[str, List] = {}
    for mask_path in annotation_files:
        image_id = get_image_id(mask_path)
        if image_id in index:
            raise ValueError(
                f"Duplicate image ID {image_id!r}. Point the script at one split/folder "
                "at a time, or adjust the image-ID generation."
            )
        index[image_id] = extract_class_ids(mask_path, ignored, id_to_name)

    return dict(sorted(index.items()))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a JSON mapping from COCO-Stuff image IDs to classes present."
    )
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        required=True,
        help="Directory of stuffthingmaps PNGs; subdirectories are searched recursively.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("cocostuff_classes.json"),
        help="Output JSON file (default: cocostuff_classes.json).",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Optional ID->name map (COCO-Stuff labels.txt or a JSON). "
        "Omit to emit numeric category IDs.",
    )
    parser.add_argument(
        "--ignore-labels",
        type=int,
        nargs="*",
        default=[255],
        help="Pixel label IDs to omit (default: 255, the 'unlabeled' void value in "
        "the official stuffthingmaps).",
    )
    args = parser.parse_args()

    id_to_name = load_label_names(args.labels)
    index = build_index(args.annotations_dir, args.ignore_labels, id_to_name)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(index, file, indent=2, ensure_ascii=False)
        file.write("\n")

    print(f"Indexed {len(index)} annotations -> {args.output}")


if __name__ == "__main__":
    main()