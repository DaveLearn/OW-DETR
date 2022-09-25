# Copyright David Pershouse 2022

# %%
from typing import Callable
from .fiftyone_datasets import get_coco_2017_train, get_coco_2017_val, known_label_subset, label_subset, random_sample, class_minimum_sample
#from fiftyone.utils.random import random_split
from fiftyone.utils.splits import random_split
import fiftyone as fo

#####
# Improved dataset and class mappings as proposed by OW-DETR paper
#


COCOIFY_MAPPING = {
    "aeroplane": "airplane",
    "diningtable":  "dining table",
    "motorbike":  "motorcycle",
    "pottedplant": "potted plant",
    "sofa": "couch",
    "tvmonitor": "tv"
}

T1_CLASS_NAMES = [
    "airplane","bicycle","bird","boat","bus","car",
    "cat","cow","dog","horse","motorcycle","sheep","train",
    "elephant","bear","zebra","giraffe","truck","person"
]

T2_CLASS_NAMES = [
    "traffic light","fire hydrant","stop sign",
    "parking meter","bench","chair","dining table",
    "potted plant","backpack","umbrella","handbag",
    "tie","suitcase","microwave","oven","toaster","sink",
    "refrigerator","bed","toilet","couch"
]

T3_CLASS_NAMES = [
    "frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard",
    "surfboard","tennis racket","banana","apple","sandwich",
    "orange","broccoli","carrot","hot dog","pizza","donut","cake"
]

T4_CLASS_NAMES = [
    "laptop","mouse","remote","keyboard","cell phone","book",
    "clock","vase","scissors","teddy bear","hair drier","toothbrush",
    "wine glass","cup","fork","knife","spoon","bowl","tv","bottle"
]

ALL_CLASS_NAMES = T1_CLASS_NAMES + T2_CLASS_NAMES + T3_CLASS_NAMES + T4_CLASS_NAMES

UNKNOWN_CLASS = "unknown"

train_val_seed = 56
train_val_split = { "train": 0.95, "val": 0.05 }

def get_owdetr_trainval():
    ds_name = 'owdetr-voc-coco-trainval'
    if fo.dataset_exists(ds_name):
        combined = fo.load_dataset(ds_name)
    else:
        combined = get_coco_2017_train().clone(name=ds_name)
     #   combined.merge_samples(get_voc_2012_train().map_labels("ground_truth", COCOIFY_MAPPING), include_info=True)
     #   combined.merge_samples(get_voc_2012_val().map_labels("ground_truth", COCOIFY_MAPPING), include_info=True)
     #   combined.merge_samples(get_voc_2007_train().map_labels("ground_truth", COCOIFY_MAPPING), include_info=True)
     #   combined.merge_samples(get_voc_2007_val().map_labels("ground_truth", COCOIFY_MAPPING), include_info=True)
        combined.persistent = True
        random_split(combined, train_val_split, seed=train_val_seed)
        combined.default_classes = ALL_CLASS_NAMES
        combined.save()

    return label_subset(combined, ALL_CLASS_NAMES)

def get_owdetr_full_val():
    ds_name = 'owdetr-voc-coco-val'
    if fo.dataset_exists(ds_name):
        combined = fo.load_dataset(ds_name)
    else:
        combined = get_owdetr_trainval().match_tags("val").clone(name=ds_name)
        combined.default_classes = ALL_CLASS_NAMES
        combined.persistent = True
        combined.save()

    return label_subset(combined, ALL_CLASS_NAMES)




def get_owdetr_full_test():
    ds_name = 'owdetr-coco-test'
    if fo.dataset_exists(ds_name):
        combined = fo.load_dataset(ds_name)
    else:
        combined = get_coco_2017_val().clone(name=ds_name)
        combined.default_classes = ALL_CLASS_NAMES
        combined.save()
    return label_subset(combined, ALL_CLASS_NAMES)

# %%
def get_owdetr_eval_trainval_t1():
    return label_subset(get_owdetr_trainval(), T1_CLASS_NAMES)


def get_owdetr_eval_trainval_t2():
    return label_subset(get_owdetr_trainval(), T2_CLASS_NAMES)


def get_owdetr_eval_trainval_t3():
    return label_subset(get_owdetr_trainval(), T3_CLASS_NAMES)


def get_owdetr_eval_trainval_t4():
    return label_subset(get_owdetr_trainval(), T4_CLASS_NAMES)




def get_owdetr_eval_train_t1():
    return label_subset(get_owdetr_trainval().match_tags("train"), T1_CLASS_NAMES)


def get_owdetr_eval_train_t2():
    return label_subset(get_owdetr_trainval().match_tags("train"), T2_CLASS_NAMES)


def get_owdetr_eval_train_t3():
    return label_subset(get_owdetr_trainval().match_tags("train"), T3_CLASS_NAMES)


def get_owdetr_eval_train_t4():
    return label_subset(get_owdetr_trainval().match_tags("train"), T4_CLASS_NAMES)




fine_tune_seed = 77

def get_owdetr_eval_ft_t1(class_min=50):
    ds_name = f'owdetr_eval_ft{class_min}_t1'

    if fo.dataset_exists(ds_name):
        return fo.load_dataset(ds_name)
    
    combined = class_minimum_sample(get_owdetr_eval_train_t1(), class_min, seed=fine_tune_seed).clone(name=ds_name)
    combined.persistent = True
    combined.save()
    return combined

def get_owdetr_eval_ft_t2(class_min=50):
    ds_name = f'owdetr_eval_ft{class_min}_t2'

    if fo.dataset_exists(ds_name):
        return fo.load_dataset(ds_name)

    combined = get_owdetr_eval_ft_t1(class_min).clone(name=ds_name)

    task2_ft = class_minimum_sample(get_owdetr_eval_train_t2(), class_min, seed=fine_tune_seed)
    combined.merge_samples(task2_ft, include_info=True)
    combined.default_classes = T1_CLASS_NAMES + T2_CLASS_NAMES
    combined.persistent = True
    combined.save()

    return combined

def get_owdetr_eval_ft_t3(class_min=50):
    ds_name = f'owdetr_eval_ft{class_min}_t3'

    if fo.dataset_exists(ds_name):
        return fo.load_dataset(ds_name)

    combined = get_owdetr_eval_ft_t2(class_min).clone(name=ds_name)

    task3_ft = class_minimum_sample(get_owdetr_eval_train_t3(), class_min, seed=fine_tune_seed)
    combined.merge_samples(task3_ft, include_info=True)
    combined.default_classes = T1_CLASS_NAMES + T2_CLASS_NAMES + T3_CLASS_NAMES
    combined.persistent = True
    combined.save()

    return combined

def get_owdetr_eval_ft_t4(class_min=50):
    ds_name = f'owdetr_eval_ft{class_min}_t4'

    if fo.dataset_exists(ds_name):
        return fo.load_dataset(ds_name)

    combined = get_owdetr_eval_ft_t3(class_min).clone(name=ds_name)

    task3_ft = class_minimum_sample(get_owdetr_eval_train_t4(), class_min, seed=fine_tune_seed)
    combined.merge_samples(task3_ft, include_info=True)
    combined.default_classes = T1_CLASS_NAMES + T2_CLASS_NAMES + T3_CLASS_NAMES + T4_CLASS_NAMES
    combined.persistent = True
    combined.save()

    return combined


# %%
def get_owdetr_eval_test_t1():
    return known_label_subset(get_owdetr_full_test(), T1_CLASS_NAMES)


def get_owdetr_eval_test_t2():
    return known_label_subset(get_owdetr_full_test(), T1_CLASS_NAMES + T2_CLASS_NAMES)


def get_owdetr_eval_test_t3():
    return known_label_subset(get_owdetr_full_test(), T1_CLASS_NAMES + T2_CLASS_NAMES + T3_CLASS_NAMES)


def get_owdetr_eval_test_t4():
    return known_label_subset(get_owdetr_full_test(), T1_CLASS_NAMES + T2_CLASS_NAMES + T3_CLASS_NAMES + T4_CLASS_NAMES)



def get_owdetr_eval_val_t1():
    return known_label_subset(get_owdetr_full_val(), T1_CLASS_NAMES)


def get_owdetr_eval_val_t2():
    return known_label_subset(get_owdetr_full_val(), T1_CLASS_NAMES + T2_CLASS_NAMES)


def get_owdetr_eval_val_t3():
    return known_label_subset(get_owdetr_full_val(), T1_CLASS_NAMES + T2_CLASS_NAMES + T3_CLASS_NAMES)


def get_owdetr_eval_val_t4():
    return known_label_subset(get_owdetr_full_val(), T1_CLASS_NAMES + T2_CLASS_NAMES + T3_CLASS_NAMES + T4_CLASS_NAMES)

datasets: dict[str, Callable[[], fo.DatasetView]] = {}

def register_fiftyone_dataset(name: str, get: Callable[[], fo.DatasetView]):
    datasets[name] = get

def get_fiftyone_dataset(name: str) -> fo.DatasetView:
    if not name in datasets:
        raise RuntimeError(f"Couldn't find {name} in registered datasets")
    return datasets[name]()

def register_owdetr_datasets():
    # %%
    mini_sample_size=3000
    mini_test_sample_size=1000
    mini_sample_seed=20
    smoke_sample_size=10

    for task in range(1,5):
        for stage in ['train', 'test', 'val', 'trainval']:
            loader = globals()[f"get_owdetr_eval_{stage}_t{task}"]
            register_fiftyone_dataset(f"owdetr_eval_{stage}_t{task}", lambda l=loader: l())
            # register mini and smoke options
            if stage == 'train':
                register_fiftyone_dataset(f"owdetr_eval_train_t{task}_mini", lambda l=loader: random_sample(l(), mini_sample_size, mini_sample_seed))
                register_fiftyone_dataset(f"owdetr_eval_train_t{task}_smoke", lambda l=loader: random_sample(l(), smoke_sample_size, mini_sample_seed))
            elif stage == 'trainval':
                register_fiftyone_dataset(f"owdetr_eval_trainval_t{task}_mini", lambda l=loader: random_sample(l(), mini_sample_size, mini_sample_seed))
                register_fiftyone_dataset(f"owdetr_eval_trainval_t{task}_smoke", lambda l=loader: random_sample(l(), smoke_sample_size, mini_sample_seed))
            elif stage == 'test':
                register_fiftyone_dataset(f"owdetr_eval_test_t{task}_mini", lambda l=loader: random_sample(l(), mini_test_sample_size, mini_sample_seed))
                register_fiftyone_dataset(f"owdetr_eval_test_t{task}_smoke", lambda l=loader: random_sample(l(), smoke_sample_size, mini_sample_seed))
            elif stage == 'val':
                register_fiftyone_dataset(f"owdetr_eval_val_t{task}_mini", lambda l=loader: random_sample(l(), mini_test_sample_size, mini_sample_seed))
                register_fiftyone_dataset(f"owdetr_eval_val_t{task}_smoke", lambda l=loader: random_sample(l(), smoke_sample_size, mini_sample_seed))


        ftloader = globals()[f"get_owdetr_eval_ft_t{task}"]
        register_fiftyone_dataset(f"owdetr_eval_ft_t{task}", lambda l=ftloader: l(class_min=100))
        register_fiftyone_dataset(f"owdetr_eval_ft_t{task}_smoke", lambda l=ftloader: random_sample(l(class_min=100), smoke_sample_size, mini_sample_seed))
        register_fiftyone_dataset(f"owdetr_eval_ftonly_t{task}", lambda l=ftloader: l(class_min=300))
        register_fiftyone_dataset(f"owdetr_eval_ftonly_t{task}_smoke", lambda l=ftloader: random_sample(l(class_min=300), smoke_sample_size, mini_sample_seed))