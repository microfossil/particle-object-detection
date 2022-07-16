"""
Convert to cocoformat
"""
import json

from visage.cvat.project_api import CVAT
from visage.project.project import Project


def project_to_coco(project: Project, labels):
    info = {}
    licences = []
    images = []
    annotations = []
    categories = []

    category_id = 1
    for label in labels:
        coco_category = {}
        coco_category["supercategory"] = "marine"
        coco_category["id"] = category_id
        coco_category["name"] = label
        categories.append(coco_category)
        category_id += 1

    image_id = 1
    annotation_id = 1
    for key, metadata in project.image_dict.items():
        coco_image_data = {}
        coco_image_data["filename"] = metadata.filename
        coco_image_data["id"] = image_id

        for ann in metadata.annotations:
            coco_annotation = {}
            coco_annotation["image_id"] = image_id
            coco_annotation["category_id"] = labels.index(ann.label) + 1
            coco_annotation["id"] = annotation_id
            coco_annotation["bbox"] = [ann.x, ann.y, ann.width, ann.height]
            coco_annotation["score"] = ann.score
            coco_annotation["segmentation"] = []
            coco_annotation["iscrowd"] = 0
            coco_annotation["area"] = ann.width * ann.height
            annotations.append(coco_annotation)
            annotation_id += 1

        images.append(coco_image_data)
        image_id += 1

    coco = {
        "info": info,
        "licences": licences,
        "categories": categories,
        "images": images,
        "annotations": annotations
    }

    return coco


def coco_evaluate(ref: Project, det: Project, labels=None):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import tempfile

    if labels is None:
        ref.update_label_dict_from_metadata()
        det.update_label_dict_from_metadata()
        labels = set()
        for label in ref.label_dict.keys():
            labels.add(label)
        for label in det.label_dict.keys():
            labels.add(label)
        labels = list(labels)


    # Convert project to dictionary in COCO format
    ref_coco = project_to_coco(ref, labels)
    det_coco = project_to_coco(det, labels)

    # Save to a temporary file
    ref_file = tempfile.NamedTemporaryFile("w")
    det_file = tempfile.NamedTemporaryFile("w")
    json.dump(ref_coco, ref_file)
    json.dump(det_coco, det_file)
    ref_file.flush()
    det_file.flush()

    cocoGt = COCO(ref_file.name)
    cocoDt = COCO(det_file.name)

    cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


def coco_evaluate_from_files(ref, det, labels):
    ref_project = Project.load(ref)
    det_project = Project.load(det)
    coco_evaluate(ref_project, det_project, labels)


if __name__ == "__main__":
    cvat_projects = CVAT("http://cruncher-ph.nexus.csiro.au:8080")
    cvat_projects.load()
    cvat_task = cvat_projects.load_task_by_name("Heron Island July 2020", "20200716_fitzroy_channel_west_v4")
    project = cvat_task.project
    coco_evaluate(project, project)
