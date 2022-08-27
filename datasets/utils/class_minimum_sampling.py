# Copyright David Pershouse 2022

import random
from .dataset_types import DetectronDict, DetectronJsonDataset
import numpy as np


def class_minimum_sample(
    datasetInfo: DetectronJsonDataset, min_instances: int, seed: int = 77
) -> DetectronJsonDataset:
    from collections import deque

    all_labels = datasetInfo["class_names"]
    dataset = datasetInfo["dataset"].copy()  # take a copy so we can shuffle
    store: list[deque] = [deque(maxlen=min_instances) for _ in range(len(all_labels))]

    rand_gen = random.Random()
    rand_gen.seed(seed)
    rand_gen.shuffle(dataset)

    for image in dataset:
        for instance in image["annotations"]:
            store[instance["category_id"]].append(image["image_id"])

        current_min_item_count = min([len(items) for items in store])
        if current_min_item_count >= min_instances:
            break

    selectedidset: set[str] = set()
    for item in store:
        selectedidset.update(item)

    selectedids = list(selectedidset)
    dataset_sampled: list[DetectronDict] = []

    for ds_item in dataset:
        if ds_item["image_id"] in selectedids:
            dataset_sampled.append(ds_item)

    return {"class_names": all_labels, "dataset": dataset_sampled}
