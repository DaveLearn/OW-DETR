import unittest
from datasets.coco import make_coco_transforms
from datasets.utils.dataset_ops import save_dataset_to_disk
from datasets.torchvision_datasets.open_world import OWDetection, JSONOWDetection
import argparse
import pathlib
import tempfile
import torch.testing

voc_path = pathlib.Path(__file__).parent.parent.resolve().joinpath("data", "OWDETR", "VOC2007")


class TestJsonOWDetection(unittest.TestCase):
    def test_get_detectron(self):
        args = argparse.Namespace(PREV_INTRODUCED_CLS=0, CUR_INTRODUCED_CLS=19, num_classes=81)
        ds = OWDetection(args, voc_path,years=["2007"],image_sets=["t1_train"],transforms=make_coco_transforms("t1_test"))
        self.assertIsNotNone(ds)
        jsonitem = ds.get_detectron(0)
        
        with tempfile.TemporaryDirectory(prefix="owdetr_unit_test") as td:
            output_file = pathlib.Path(td).joinpath("temp_json_dataset.json")
            save_dataset_to_disk({ "dataset": [ds.get_detectron(i) for i in range(0, len(ds))], "class_names": list(ds.CLASS_NAMES)}, output_file)

            json_ds = JSONOWDetection(td, "temp_json_dataset", transforms=make_coco_transforms("t1_test"))

            for i in range(0, len(ds)):
                from_json_file, from_json_dict = json_ds.get_raw(i)
                from_ds_file, from_ds_dict = ds.get_raw(i)

                self.assertEqual(from_ds_file, from_json_file)
                torch.testing.assert_close(from_ds_dict["area"], from_json_dict["area"])
                torch.testing.assert_close(from_ds_dict["boxes"], from_json_dict["boxes"])
                torch.testing.assert_close(from_ds_dict["labels"], from_json_dict["labels"])
                torch.testing.assert_close(from_ds_dict["image_id"], from_json_dict["image_id"])
                torch.testing.assert_close(from_ds_dict["orig_size"], from_json_dict["orig_size"])
                torch.testing.assert_close(from_ds_dict["size"], from_json_dict["size"])