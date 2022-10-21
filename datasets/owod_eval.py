# Based on orignal detectron2 pascal_voc_evaluation.py which is Copyright (c) Facebook, Inc. and its affiliates.
# Including changes made by the OWOD ORE paper
# And additions made by David Pershouse to improve the metrics

import logging
from typing import Any, Union, cast
import numpy as np
from collections import defaultdict
import torch
import json
import os

from datasets.torchvision_datasets.open_world import JSONOWDetection, OWDatasetDict, OWDetection
from util.misc import all_gather, get_sha

class OWODEvaluator():
    """
    Evaluate OWOD dataset using VOC style AP.
    It contains a synchronization, therefore has to be called from all ranks.

    Note that the concept of AP can be implemented in different ways and may not
    produce identical results. This class mimics the implementation of the official
    Pascal VOC Matlab API, and should produce similar but not identical results to the
    official API.
    """

    def __init__(self, test_ds: Union[JSONOWDetection, OWDetection], args: Any):
        """
        Args:
            dataset_name (str): name of the dataset, e.g., "voc_2007_test"
        """
        self.ds = test_ds
        self.class_name_idx = test_ds.CLASS_NAMES
        
        #meta = MetadataCatalog.get(dataset_name)

        # Too many tiny files, download all to local for speed.
        # annotation_dir_local = PathManager.get_local_path(
        #    os.path.join(meta.dirname, "Annotations/")
        #)
        #self._anno_file_template = os.path.join(annotation_dir_local, "{}.xml")
        #self._image_set_path = os.path.join(meta.dirname, "ImageSets", "Main", meta.split + ".txt")
        self._class_names = test_ds.CLASS_NAMES
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
       
        self.img_ids = []
        self.lines = []
        self.lines_cls = []

        self.summary = defaultdict()
        
        # the caller expects this structure
        self.coco_eval = dict(bbox=lambda: None)
        self.coco_eval['bbox'].stats = torch.tensor([])
        self.coco_eval['bbox'].eval = dict()

        if args is not None:
            self.prev_intro_cls = args.PREV_INTRODUCED_CLS
            self.curr_intro_cls = args.CUR_INTRODUCED_CLS
            self.total_num_class =  args.num_classes
            self.unknown_class_index = self.total_num_class - 1
            self.num_seen_classes = self.prev_intro_cls + self.curr_intro_cls
            self.output_dir = args.output_dir

    def update(self, predictions):
        for img_id, pred in predictions.items():
            image_id = img_id
            pred_boxes, pred_labels, pred_scores = [pred[k].cpu() for k in ['boxes', 'labels', 'scores']]
            classes = pred_labels.tolist()
            for (xmin, ymin, xmax, ymax), cls, score in zip(pred_boxes.tolist(), classes , pred_scores.tolist()):
                xmin += 1
                ymin += 1 # our stored dataset has already done the subtraction``
                self.lines.append(f"{image_id} {score:.3f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}")
                self.lines_cls.append(cls)


    def synchronize_between_processes(self):
        self.img_ids = torch.tensor(self.img_ids, dtype=torch.int64)
        self.lines_cls = torch.tensor(self.lines_cls, dtype=torch.int64)
        self.img_ids, self.lines, self.lines_cls = self.merge(self.img_ids, self.lines, self.lines_cls)

    def merge(self, img_ids, lines, lines_cls):
        flatten = lambda ls: [s for l in ls for s in l]

        all_img_ids = torch.cat(all_gather(img_ids))
        all_lines_cls = torch.cat(all_gather(lines_cls))
        all_lines = flatten(all_gather(lines))
        return all_img_ids, all_lines, all_lines_cls

    # below two methods taken from JosephKJ/owod repository MIT licensed. 
    def compute_WI_at_many_recall_level(self, recalls, tp_plus_fp_cs, fp_os):
        wi_at_recall = {}
        for r in range(1, 10):
            r = r/10
            wi = self.compute_WI_at_a_recall_level(recalls, tp_plus_fp_cs, fp_os, recall_level=r)
            wi_at_recall[r] = wi
        return wi_at_recall

    def compute_WI_at_a_recall_level(self, recalls, tp_plus_fp_cs, fp_os, recall_level=0.5):
        wi_at_iou = {}
        for iou, recall in recalls.items():
            tp_plus_fps = []
            fps = []
            for cls_id, rec in enumerate(recall):
                # our test datasets only include classes we have seen or unknown. We skip unknown (cls_id 0) by starting from 1
                if cls_id in range(self.num_seen_classes) and len(rec) > 0:
                    index = min(range(len(rec)), key=lambda i: abs(rec[i] - recall_level))
                    tp_plus_fp = tp_plus_fp_cs[iou][cls_id][index]
                    tp_plus_fps.append(tp_plus_fp)
                    fp = fp_os[iou][cls_id][index]
                    fps.append(fp)
            if len(tp_plus_fps) > 0:
                wi_at_iou[iou] = np.mean(fps) / np.mean(tp_plus_fps)
            else:
                wi_at_iou[iou] = 0
        return wi_at_iou

    def accumulate(self):
        """
        Returns:
            dict: has a key "segm", whose value is a dict of "AP", "AP50", and "AP75".
        """
        self._logger.info(
            "Evaluating using {} metric. "
            "Note that results do not use the official Matlab API.".format(
                 2012
            )
        )

        # load annotationss from dataset once
        ds = self.ds
        ds_class_names = self._class_names

        instances = {}
        anno: OWDatasetDict
        for idx in range(len(ds)):
            _, anno = ds.get_raw(idx)
            instances[str(anno['image_id'].item())] = [ 
                {
                    "name":  ds_class_names[label],
                    "bbox": [int(box[0] + 1), int(box[1] + 1), int(box[2]), int(box[3])], # add on one to our GT to match the voc annotations
                    "difficult": False # force false since if we wanted to skip evaluation on them, then they should be filtered at the owod dataset level
                }
                for label, box in zip(anno["labels"].tolist(), anno["boxes"].tolist())
            ]

        ds = None

       
        aps = defaultdict(list)  # iou -> ap per class
        aps07 = defaultdict(list) # iou -> ap voc2007 metric per class

        # Extra owod stats
        recs = defaultdict(list)  # recall
        precs = defaultdict(list)  # precision
        all_recs = defaultdict(list) 
        all_precs = defaultdict(list)
        unk_det_as_knowns = defaultdict(list)
        num_unks = defaultdict(list)
        tp_plus_fp_cs = defaultdict(list)
        tpos = defaultdict(list)
        fpos = defaultdict(list)
        fp_os = defaultdict(list)
        fp_known = defaultdict(list)

        for class_label_ind, cls_name in enumerate(self._class_names):

            lines_by_class = [l + '\n' for l, c in zip(self.lines, self.lines_cls.tolist()) if c == class_label_ind]

            # for thresh in range(50, 100, 5):
            thresh = 50 # OWOD eval only does threshold of 50
            rec, prec, ap, unk_det_as_known, num_unk, tp, fp, fp_open_set, fp_k, ap07 = voc_eval(
                lines_by_class,
                instances,
                cls_name,
                ovthresh=thresh / 100.0              
            )
            aps[thresh].append(ap * 100)
            aps07[thresh].append(ap07 * 100)
            unk_det_as_knowns[thresh].append(unk_det_as_known)
            num_unks[thresh].append(num_unk)
            all_precs[thresh].append(prec)
            all_recs[thresh].append(rec)
            tpos[thresh].append(tp)
            fpos[thresh].append(fp)
            tp_plus_fp_cs[thresh].append(tp+fp)
            fp_os[thresh].append(fp_open_set)
            fp_known[thresh].append(fp_k)
            try:
                recs[thresh].append(rec[-1] * 100)
                precs[thresh].append(prec[-1] * 100)
            except:
                recs[thresh].append(0)
                precs[thresh].append(0)

     

        wi = self.compute_WI_at_many_recall_level(all_recs, tp_plus_fp_cs, fp_os)
        total_num_unk_det_as_known = {iou: int(np.sum(x)) for iou, x in unk_det_as_knowns.items()}
        total_num_unk = num_unks[50][0]
         
        # UDR and UDP from Revisiting Open World Object Detection (Zhao et al 2022)

        # Unknown Detection Recall (UDR) is accurate localization rate of unknown classes (Unknown objectness, distinguishing an unknown instance from background)
        # UDR = (True Positive Unknown (unknowns boxes detected as unknown) + (Unknown boxes detected as known class))  /  True Positive Unknown + False Negative Unknown
        # or more simply
        # UDR = Number of unknown instances detected as non background / total number of ground truth unknown instances
        # It is like the regular Recall for unknown except that false negatives that end up labelled as another known class are considered as true positives too (ignores classification accuracy)
        # Percentage of unknowns correctly localized

        # unknown detected as unknown
        tp_unknown = tpos[50][self.unknown_class_index][-1] if len(tpos[50][self.unknown_class_index]) > 0 else 0
        # unknown detected as other
        fn_unknown_star = total_num_unk_det_as_known[50]

        localized_unknowns = (tp_unknown + fn_unknown_star)

        if total_num_unk > 0:
            udr = (localized_unknowns / total_num_unk) * 100
        else:
            udr = 100.0  # 100 percent of all unknowns were localized

        # Unknown detection Precision (UDP) is accurate classification rate of all localized unknown instances (distinguish an unknown instance from similar known class)
        # independent of recall, how good is it at distinguishing between known and unknown.
        # UDP = True Positive Unknown (unknown boxes detected as unknown) / True Positive Unknown + (Unknown boxes detected as anything)
        # or more simply
        # UDP = number of unknown instances detected as unknown /  number of unknown instances detected as non background
        
        if localized_unknowns > 0:
            udp = (tp_unknown / localized_unknowns) * 100
        else:
            udp = 100.0 # 100 percent of all localized unknowns were classified correctly

        # UDR is like "unknown localization recall" (percent of unknown objects localized) and UDP is more like "unknown classification recall" (percent of localized unknown objects classified as unknown)
        # it isn't really precision since it doesn't penalize false positives. In reality UDP * UDR = U-Recall, we should call it ULR and UCR (unknown localization recall and unknown classification recall)

        ulr = udr
        ucr = udp

        # we can propose a similar "precision" metric
        # Unknown Localization Precision (ULP) = Number of non background instances detected as unknown / total number of detections = TP + FP_known (false positives overlapping with known GT) / TP + FP_known + FP_background
        # and then Unknown Classification Precision (UCP) = Number of unknown instances detected / Number of non background instances detected as unknown = TP / TP + FP_Known
        
        total_num_known_det_as_unknown = fp_known[50][self.unknown_class_index][-1] if len(fp_known[50][self.unknown_class_index]) > 0 else 0
        fp_unknown = fpos[50][self.unknown_class_index][-1] if len(fpos[50][self.unknown_class_index] > 0) else 0

        if (tp_unknown + fp_unknown) > 0:
            ulp = (tp_unknown + total_num_known_det_as_unknown) / (tp_unknown + fp_unknown)
        else:
            ulp = 0.0 # 0% of our non existent guesses lined up with ground truth

        if (tp_unknown + total_num_known_det_as_unknown) > 0:
            ucp = tp_unknown / (tp_unknown + total_num_known_det_as_unknown)
        else:
            ucp = 0.0 # 0% of our non existent guesses lined up with ground truth

        prev_known = self.prev_intro_cls

        ret = {}
        # mAP = {iou: np.mean(x) for iou, x in aps.items()}
        ret["git-sha"] = get_sha()
        ret["AP50-Known"] = np.mean(aps[50][0:self.num_seen_classes])
        ret["AP50-07-Known"] = np.mean(aps07[50][0:self.num_seen_classes])

        ret["Prev-Known-Classes"] = prev_known
        if prev_known is not None and prev_known > 0:
            ret["AP50-Prev"] = np.mean(aps[50][0:prev_known])
            ret["AP50-Curr"] = np.mean(aps[50][prev_known:self.num_seen_classes])
            ret["AP50-07-Prev"] = np.mean(aps07[50][0:prev_known])
            ret["AP50-07-Curr"] = np.mean(aps07[50][prev_known:self.num_seen_classes])
        else:
            ret["AP50-Prev"] = 0
            ret["AP50-Curr"] = ret["AP50-Known"]
            ret["AP50-07-Prev"] = 0
            ret["AP50-07-Curr"] = ret["AP50-07-Known"]

        ret["F1-i"] = (2 * ret["AP50-Prev"] * ret["AP50-Curr"]) / (ret["AP50-Prev"] + ret["AP50-Curr"])

        ret["WI"] = { iou: widx[50] for iou, widx in wi.items() }
        ret["ULR-UDR"] = ulr
        ret["UCR-UDP"] = ucr
        ret["TP-Unknown"] = tp_unknown
        ret["FP-Known-as-Unknown"] = total_num_known_det_as_unknown
        ret["FP-Unknown-as-Known"] = total_num_unk_det_as_known[50]
        ret["FP-Unknown-as-Known-By-Class"] = unk_det_as_knowns[50]
        ret["FP-Unknown"] = fp_unknown
        ret["ULP"] = ulp
        ret["UCP"] = ucp
        ret["A-OSE"] = total_num_unk_det_as_known[50]
        ret["Unknown"] = int(total_num_unk)
        ret["Classes"] = self._class_names
        ret["AP50"] = aps[50]
        ret["Precision50"] = precs[50]
        ret["Recall50"] = recs[50]
        ret["Summary"] = f"AP Known (prev/curr): {ret['AP50-Known']:.2f} ({ret['AP50-Prev']:.2f}/{ret['AP50-Curr']:.2f}), F1-i: {ret['F1-i']:.2f}, URec: {recs[50][0]:.2f}, UPre: {precs[50][0]:.2f}, WI 0.8: {wi[0.8][50]:.5f}, A-OSE: {ret['A-OSE']}, AP Known 07 (prev/curr): {ret['AP50-07-Known']:.2f} ({ret['AP50-07-Prev']:.2f}/{ret['AP50-07-Curr']:.2f})"
        # ret["Summary"] = f" AP Known: {ret['AP50-Known']:.2f}, URec: {recs[50][self.unknown_class_index]:.2f}, ULR/UCR(UDR/UDP): {udr:.2f} / {udp:.2f},  UPre: {precs[50][self.unknown_class_index]:.2f}, ULP/UCP: {ulp:.2f} / {ucp:.2f}, WI 0.8: {wi[0.8][50]:.5f}, A-OSE: {ret['A-OSE']}"
        self.summary = ret

    def summarize(self):
        self._logger.info(f"Result summary : {self.summary}")
        print(f"Result Summary: {self.summary}")

        if self.output_dir is not None: 
            dump_file = os.path.join(self.output_dir, "results.json")
            with open(dump_file, "w") as fd:
                json.dump(self.summary, fd, indent=4 ) 

##############################################################################
#
# Below code is modified from
# https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py
# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
#
# Modified to accept a detectron2 dataset name instead of text files in voc format by David Pershouse
# --------------------------------------------------------

"""Python implementation of the PASCAL VOC devkit's AP evaluation code."""


def voc_ap(rec: list[float], prec: list[float], use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = cast(int, np.where(mrec[1:] != mrec[:-1])[0])

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap



def voc_eval(detlines, recs, classname, ovthresh=0.5):
    """rec, prec, ap = voc_eval(detpath,
                                dataset_name,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    recs: dictionary of image_id: { name, bbox, difficult } containing all images
    classname: Category name (duh)
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in recs:
        R = [obj for obj in recs[imagename] if obj["name"] == classname]
        bbox = np.array([x["bbox"] for x in R])
        difficult = np.array([x["difficult"] for x in R]).astype(np.bool8)
        # difficult = np.array([False for x in R]).astype(np.bool)  # treat all "difficult" as GT
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det}

    known_class_recs = None
    if classname == "unknown":
        known_class_recs = {}
        n_known = 0
        for imagename in recs:
            RK = [obj for obj in recs[imagename] if obj["name"] != classname]
            bbox = np.array([x["bbox"] for x in RK])
            difficult = np.array([x["difficult"] for x in RK]).astype(np.bool8)
            # difficult = np.array([False for x in R]).astype(np.bool)  # treat all "difficult" as GT
            det = [False] * len(RK)
            n_known = n_known + sum(~difficult)
            known_class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det}




    lines = detlines

    splitlines = [x.strip().split(" ") for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines]).reshape(-1, 4)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    fp_known = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        jmax = -1
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih

            # union
            uni = (
                (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
                - inters
            )

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R["difficult"][jmax]:
                if not R["det"][jmax]:
                    tp[d] = 1.0
                    R["det"][jmax] = 1
                else:
                    fp[d] = 1.0
        else:
            fp[d] = 1.0

        # see if this unknown false positive detection overlaps with known class
        # (cut and paste is a little disgusting, but this is less impact to original code path)
        if classname == "unknown" and fp[d] > 0 and known_class_recs is not None:
            RK = known_class_recs[image_ids[d]]
            ovmax = -np.inf
            BBGT = RK["bbox"].astype(float)

            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
                ih = np.maximum(iymax - iymin + 1.0, 0.0)
                inters = iw * ih

                # union
                uni = (
                    (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                    + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
                    - inters
                )

                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not RK["difficult"][jmax]:
                    if not RK["det"][jmax]:
                        fp_known[d] = 1.0
                        RK["det"][jmax] = 1

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    fp_known = np.cumsum(fp_known)

    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(cast(list[float], rec), cast(list[float], prec), use_07_metric=False)
    ap07 = voc_ap(cast(list[float], rec), cast(list[float], prec), use_07_metric=True) # report 07 to match owdetr eval implementation
    '''
    Computing Absolute Open-Set Error (A-OSE) and Wilderness Impact (WI)
                                    ===========    
    Absolute OSE = # of unknown objects classified as known objects of class 'classname'
    WI = FP_openset / (TP_closed_set + FP_closed_set)
    '''
    logger = logging.getLogger(__name__)

    # Finding GT of unknown objects
    unknown_class_recs = {}
    n_unk = 0
    for imagename in recs:
        R = [obj for obj in recs[imagename] if obj["name"] == 'unknown']
        bbox = np.array([x["bbox"] for x in R])
        difficult = np.array([x["difficult"] for x in R]).astype(np.bool8)
        det = [False] * len(R)
        n_unk = n_unk + sum(~difficult)
        unknown_class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det}

    if classname == 'unknown':
        return rec, prec, ap, 0, n_unk, tp, fp, None, fp_known, ap07

    # Go down each detection and see if it has an overlap with an unknown object.
    # If so, it is an unknown object that was classified as known.
    is_unk = np.zeros(nd)
    for d in range(nd):
        R = unknown_class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih

            # union
            uni = (
                (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
                - inters
            )

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            is_unk[d] = 1.0

    is_unk_sum = np.sum(is_unk)
    # OSE = is_unk / n_unk
    # logger.info('Number of unknowns detected knowns (for class '+ classname + ') is ' + str(is_unk))
    # logger.info("Num of unknown instances: " + str(n_unk))
    # logger.info('OSE: ' + str(OSE))

    #tp_plus_fp_closed_set = tp+fp
    fp_open_set = np.cumsum(is_unk)

    return rec, prec, ap, is_unk_sum, n_unk, tp, fp, fp_open_set, fp_known, ap07


   