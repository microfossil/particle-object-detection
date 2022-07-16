import os
from collections import OrderedDict

import click

from visage.inference.evaluate import evaluate_from_files
from visage.project.image import ImageMetadata

from visage.inference.object_detection import ObjectDetector
from visage.project.project import Project


@click.group()
def cli():
    pass


@cli.command()
@click.option('--input', type=str,
              prompt='project to perform inference on',
              help='Path to project')
@click.option('--output_dir', type=str,
              prompt='directory to save projects',
              help='Path to save project')
@click.option('--name', type=str,
              prompt='name of project to save',
              help='Path to save project')
@click.option('--model', type=str,
              prompt='path to saved model',
              help='Path to saved_model directory')
@click.option('--label', type=str, multiple=True,
              help='Labels in model')
@click.option('--threshold', type=float, default=0.1,
              help='Confidence threshold')
@click.option('--skip', type=int, default=1,
              help='Number of files to skip when performing inference')
@click.option('--gpu', type=int, default=0,
              help='which GPU to use')
def detect(input, output_dir, name, model, label, threshold, skip, gpu):
    det = ObjectDetector(model, cls_labels=['background', *label], gpu=gpu)
    if os.path.isdir(input):
        project = det.detect_directory(input,
                                       threshold,
                                       skip,
                                       use_tracker=True)
    else:
        project = det.detect_project_file(input,
                                          threshold,
                                          skip,
                                          use_tracker=True)
    project.metadata["name"] = name
    project.save_as(os.path.join(output_dir, name + ".json"))
    project.save_as(os.path.join(output_dir, name + ".csv"), format="mlboxv2")
    project.save_as(os.path.join(output_dir, name + "_via.json"), "via")


@cli.command()
@click.option('--ref', type=str,
              prompt='reference project',
              help='Path to reference project')
@click.option('--det', type=str,
              prompt='detections project',
              help='Path to detections project')
@click.option('--threshold', type=float, default=0.2,
              help='Confidence threshold')
def evaluate(ref, det, threshold):
    evaluate_from_files(ref, det, threshold)

#
# @cli.command()
# @click.option('--input', type=str,
#               prompt='project to clean up',
#               help='Path to project')
# @click.option('--min_seq', type=int, default=1,
#               help='Number of files to skip when performing inference')
# @click.option('--remove_edge', type=int, default=0,
#               help='Remove annotations on the edge')
# def cleanup(input, min_seq, remove_edge):
#     project = Project.load(input)
#     if remove_edge > 0:
#         new_project = project.copy()
#         new_project.image_dict = OrderedDict()
#         for key, im in project.image_dict.items():
#             new_im = ImageMetadata(im.filename)
#             for ann in im.annotations:
#                 if ann.x < remove_edge \
#                     or ann.y < remove_edge \
#                     or ann.x + ann.width > 2448 - remove_edge \
#                     or ann.y + ann.height > 2048 - remove_edge:
#                     continue
#                 else:
#                     new_im.add_annotation(ann)
#             new_project.add_image(new_im)
#     else:
#         new_project = project.copy()
#     new_project.update_label("COTS", "COTS_NV")
#     new_project.sequence_annotations(time_from_filename=True)
#     new_project.remove_sequences_by_length(min_seq)
#     project_directory = os.path.dirname(input)
#     project_filename = os.path.splitext(os.path.basename(input))[0]
#     project_filename += "_clean"
#     new_project.save_as(os.path.join(project_directory, project_filename + ".json"))
#     new_project.save_as(os.path.join(project_directory, project_filename + "_via.json"), "via")
#     new_project.save_as(os.path.join(project_directory, project_filename + ".xml"), "cvat_seq")


if __name__ == "__main__":
    cli()