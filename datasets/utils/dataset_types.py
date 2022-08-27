# Copyright David Pershouse 2022

from typing import List, TypedDict
from typing_extensions import NotRequired

class DetectronAnnotationDict(TypedDict):
    category_id: int
    bbox: List[float]
    bbox_mode: int
    category_name: NotRequired[str] 

class DetectronDict(TypedDict):
    file_name: str
    image_id: str
    height: int
    width: int
    annotations: List[DetectronAnnotationDict]

class DetectronJsonDataset(TypedDict):
    dataset: List[DetectronDict]
    class_names: List[str]