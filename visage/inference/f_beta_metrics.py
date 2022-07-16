import numpy as np
from visage.project.project import Project


def f_beta_score(ref_project: Project, det_project: Project, iou_threshold, score_threshold, beta=2):
    TP_count = 0
    FP_count = 0
    FN_count = 0
    total = 0
    for key, ref_metadata in ref_project.image_dict.items():
        # Check image is in both projects
        if key not in det_project.image_dict:
            continue
        det_metadata = det_project.image_dict[key]

        # Filter out detections below threshold
        ref_annotations = []
        for ann in ref_metadata.annotations:
            if ann.score > score_threshold:
                ref_annotations.append(ann)
        det_annotations = []
        for ann in det_metadata.annotations:
            if ann.score > score_threshold:
                det_annotations.append(ann)

        # Continue early if no annotations
        if len(ref_annotations) == 0 and len(det_annotations) == 0:
            continue

        # Sort detections by descending order of confidence score
        det_annotations = sorted(det_annotations, key=lambda x: x.score, reverse=True)

        # Compare each detection with the reference annotations to check for a match
        for det_ann in det_annotations:
            total += 1
            best_iou = iou_threshold
            matched_ann = None
            for ref_ann in ref_annotations:
                iou = det_ann.iou(ref_ann)
                if iou > best_iou:
                    matched_ann = ref_ann
                    best_iou = iou
            # If match, remove the reference annotation from the list and count as a true positive
            if matched_ann is not None:
                ref_annotations.remove(matched_ann)
                TP_count += 1
            # Otherwise, it was a false positive
            else:
                FP_count += 1

        # Any remaining reference annotations were not matched, and are counted as a false negative
        FN_count += len(ref_annotations)

    # print(total)
    # print(TP_count)
    # print(FP_count)
    # print(FN_count)

    # Return the F beta score
    try:
        F_beta = ((1 + beta ** 2) * TP_count) / ((1 + beta ** 2) * TP_count + beta ** 2 * FN_count + FP_count)
    except:
        F_beta = 0
    return F_beta, TP_count, FP_count, FN_count


def f_beta_score_from_files(ref_project_file, det_project_file, iou_threshold, score_threshold, beta=2):
    ref_project = Project.load(ref_project_file)
    det_project = Project.load(det_project_file)
    return f_beta_score(ref_project, det_project, iou_threshold, score_threshold, beta)


def f_beta_score_iou_sweep(ref_project: Project, det_project: Project, iou_thresholds, score_threshold, beta=2):
    F_beta_scores = []
    for iou_threshold in iou_thresholds:
        score = f_beta_score(ref_project, det_project, iou_threshold, score_threshold, beta)[0]
        F_beta_scores.append(score)
    return np.mean(F_beta_scores)


def f_beta_score_iou_sweep_from_files(ref_project_file, det_project_file, iou_thresholds, score_threshold, beta=2):
    F_beta_scores = []
    for iou_threshold in iou_thresholds:
        score = f_beta_score_from_files(ref_project_file, det_project_file, iou_threshold, score_threshold, beta)[0]
        F_beta_scores.append(score)
    return np.mean(F_beta_scores)


if __name__ == "__main__":
    ref_file = ""
    ref_project = Project.load(ref_file)
    det_project = Project.load(ref_file)
    print("F score should be 1.0:")
    print(f_beta_score(ref_project, det_project, 0.5, -2, beta=2))

    # Make every second value less iou
    toggle = 1
    total = 0
    t_yes = 0
    t_no = 0
    for key, metadata in det_project.image_dict.items():
        for ann in metadata.annotations:
            total += 1
            ann.width = ann.width / toggle
            toggle += 1
            if toggle >= 4:
                toggle = 1


    print(total)
    print(t_yes)
    print(t_no)

    print("F score should be close to 0.5:")
    print(f_beta_score(ref_project, det_project, 0.5, -2, beta=3))
    print(f_beta_score_iou_sweep(ref_project, det_project, [0.2, 0.5, 0.8], -2, beta=3))





