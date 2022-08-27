# Copyright David Pershouse 2022

# %%
from typing import List
import fiftyone as fo
import fiftyone.zoo as foz
import os
from fiftyone import ViewField as F
import fiftyone.core.collections as foc

fo.config.dataset_zoo_dir =  os.path.abspath(os.path.join(os.path.dirname(__file__), "../../datasets"))

# %%
def get_or_create_zoo_dataset(dataset_name: str, zoo_name: str, split: str):
    if not fo.dataset_exists(dataset_name):
        dataset=foz.load_zoo_dataset(zoo_name, split=split, dataset_name=dataset_name)
        dataset.compute_metadata()
        dataset.persistent=True
        return dataset
    else:
       return fo.load_dataset(dataset_name)

def get_voc_2007_val():
    return get_or_create_zoo_dataset("voc-2007-val", "voc-2007", "validation") 

def get_voc_2007_train():
    return get_or_create_zoo_dataset("voc-2007-train", "voc-2007", "train")

def get_voc_2012_val():
    return get_or_create_zoo_dataset("voc-2012-val", "voc-2012", "validation") 

def get_voc_2012_train():
    return get_or_create_zoo_dataset("voc-2012-train", "voc-2012", "train")

def get_coco_2017_val():
    return get_or_create_zoo_dataset("coco-2017-val", "coco-2017", "validation") 

def get_coco_2017_train():
    return get_or_create_zoo_dataset("coco-2017-train", "coco-2017", "train")

# %%
def get_voc_2007_test():
    if fo.dataset_exists('voc-2007-test'):
        return fo.load_dataset('voc-2007-test')

    url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar"
    download_dir = os.path.join(fo.config.dataset_zoo_dir, "manual-voc-2007-test")
    voc2007_dir = os.path.join(download_dir, "VOCdevkit", "VOC2007")

    if not os.path.exists(voc2007_dir):
        print(f"Downloading and extracting VOC 2007 test set to: {download_dir}")
        os.makedirs(download_dir, exist_ok=True)
        res = os.system(f"wget -c '{url}' -O - | tar -x -C '{download_dir}'")
        if res != 0 or not os.path.exists(voc2007_dir):
            raise RuntimeError(f"Couldn't download and untar VOC2007 to {download_dir}")
   
    dataset = fo.Dataset.from_dir(
        dataset_dir=voc2007_dir,
        dataset_type=fo.types.VOCDetectionDataset,
        labels_path='Annotations',
        data_path='JPEGImages',
        name='voc-2007-test')
    dataset.persistent = True
    
    return dataset

def has_at_least_one_instance_of(classes):
    return F("ground_truth.detections").filter(F("label").is_in(classes)).length() > 0

def label_subset(dataset: foc.SampleCollection, labels: List[str]) -> fo.DatasetView:
    ds = dataset.match(has_at_least_one_instance_of(labels))
    ds.default_classes = labels
    return ds.filter_labels("ground_truth", F("label").is_in(labels))

def known_label_subset(dataset: foc.SampleCollection, known_labels: List[str]) -> fo.DatasetView:
    all_labels = dataset.distinct("ground_truth.detections.label")
    unknown_labels = [ u for u in all_labels if u not in known_labels ]
    if len(unknown_labels) == 0:
        ds = dataset.view()
        ds.default_classes = all_labels
    else:
        ds = dataset.map_labels("ground_truth", { unk: 'unknown' for unk in unknown_labels })
        ds.default_classes = known_labels + [ 'unknown' ]
    return ds

def random_sample(dataset: foc.SampleCollection, sample_size, seed=77) -> fo.DatasetView:
    return dataset.take(sample_size, seed)

def class_minimum_sample(dataset: foc.SampleCollection, min_instances: int, seed=77):
    from collections import deque
    all_labels = dataset.distinct("ground_truth.detections.label")
    store: list[deque] = [deque(maxlen=min_instances) for _ in range(len(all_labels))]

    for image in dataset.shuffle(seed):
        for instance in image["ground_truth"]["detections"]:
            store[all_labels.index(instance["label"])].append(image.id)
        
        current_min_item_count = min([len(items) for items in store ])
        if current_min_item_count >= min_instances:
            break

    selectedidset: set[str] = set()
    for item in store:
        selectedidset.update(item)
    
    selectedids = list(selectedidset)
    selectedids.sort()

    return dataset.select(selectedids, ordered=True).shuffle(seed)
