import os
from pathlib import Path

import numpy as np

from visage.inference.coco_metrics import coco_evaluate
from visage.inference.f_beta_metrics import f_beta_score
from visage.project.project import Project


def evaluate(reference: Project, detections: Project, threshold):
    # Ignore threshold for COCO as it uses PR curve
    coco_evaluate(reference, detections, )
    # Do threshold for F2 score however
    detections.remove_annotations_below_threshold(threshold)
    all_f2s = []
    for iou in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.80]:
        f2 = f_beta_score(reference, detections, iou, 0.01, beta=2)
        all_f2s.append(f2[0])
        print(f"IoU: {iou:.02f}, F2 score: {f2[0]:.03f}, TP: {f2[1]}, FP: {f2[2]}, FN: {f2[3]}")
    print(f"Mean F2 score: {np.mean(all_f2s):.03f}")


def evaluate_from_files(reference_fn, detections_fn, threshold):
    reference = Project.load(reference_fn)
    detections = Project.load(detections_fn)
    evaluate(reference, detections, threshold)
    # Create comparison project
    comparison_fn = os.path.join(os.path.dirname(detections_fn), str(Path(detections_fn).stem) + "_compare_via.json")
    reference.add_suffix_to_labels("REF")
    detections.add_suffix_to_labels("DET")
    reference.add_annotations_from_project(detections, add_all_annotations=True)
    reference.save_as(comparison_fn, format="via")
