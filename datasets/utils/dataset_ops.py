# Copyright David Pershouse 2022

import json
import os
from typing import List

from .dataset_types import DetectronJsonDataset


def remap_category_ids(
    dataset: DetectronJsonDataset, class_name_idx: List[str]
) -> DetectronJsonDataset:
    dataset_cats = dataset["class_names"]
    remapped = dataset["dataset"].copy()
    for img in remapped:
        img["annotations"] = img["annotations"].copy()
        for anno in img["annotations"]:
            existing_cat = dataset_cats[anno["category_id"]]
            if existing_cat in class_name_idx:
                anno["category_id"] = class_name_idx.index(existing_cat)
            else:
                raise RuntimeError(
                    f"couldnt find remap index for {existing_cat} ({anno['category_id']}) in {class_name_idx}"
                )
    return {"dataset": remapped, "class_names": class_name_idx}

    

def save_dataset_to_disk(dataset: DetectronJsonDataset, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(dataset, f, indent="\t")


def load_dataset_from_disk(filepath) -> DetectronJsonDataset:
    with open(filepath) as file:
        ds = json.load(file)
    return ds
