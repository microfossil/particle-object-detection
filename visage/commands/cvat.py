import click

from visage.cvat.project_api import CVAT, create_task_annotations_patch
from visage.inference.converter import convert_mlboxv2_as_is
from visage.project.project import Project
from visage.cvat.archive.cli import main as cvat_main

import requests
from requests.auth import HTTPBasicAuth
import json

@click.group()
def cli1():
    pass


@cli1.command()
@click.option('--cvat_address', type=str,
              help='Address of CVAT server')
@click.option('--input', type=str,
              help='Path to detections CSV')
@click.option('--remove_from_start', type=str, default = '/mnt/ssd/',
              help='remove first part of the filename path in files referenced from detections CSV')
@click.option('--project', type=str,
              help='CVAT project name')
@click.option('--task', type=str,
              help='CVAT task name')
@click.option('--overwrite/--no-overwrite', default=False,
              help='Overwrite existing detections')
@click.option('--set_nv_label/--no-set_nv_label', default=False,
              help='if true, each label is appended with "_NV" (non validated) as per our CVAT dataset aggregation strategy')

def upload_mlboxv2_detections(cvat_address, input, remove_from_start, project, task, overwrite, set_nv_label):
    projects = CVAT(cvat_address)
    projects.load()

    # CVAT Project
    cvat_project = projects.load_project_by_name(project)

    # Load new detection project data
    new_project = convert_mlboxv2_as_is(input, remove_from_start=remove_from_start, set_nv_label=set_nv_label)

    # Show the project info
    new_project.summary()

    # Remove the start of the path as the CVAT mounted drive root / is at /mnt/ssd
    new_project.update_image_directory_start("/mnt/ssd/", "")
    new_project.update_image_directory_start("/mnt/nas/", "")

    # Create task if doesnt exist
    is_new = False
    if task not in cvat_project.task_to_id_dict.keys():
        print("Creating new task")
        is_new = True
        filenames = [v.filename for k, v in new_project.image_dict.items()]
        cvat_project.create_task(task, filenames)

    # Load the task
    print("Load task")
    cvat_task = projects.load_task_by_name(project, task)

    if not is_new:
        # Combine the annotations
        ml_project = cvat_task.project

        # Remove existing annotations so that we can do a patch
        for key, metadata in ml_project.image_dict.items():
            metadata.annotations = []

        print(f"Number of images: {len(ml_project.image_dict)}")
        ml_project.add_annotations_from_project(new_project,
                                             add_missing_images=False,
                                             add_missing_annotations=True,
                                             overwrite_annotation_dimensions=False,
                                             overwrite_annotation_label=False,
                                             add_all_annotations=True)
        print(f"Combined label count: {ml_project.label_count()}")
        print(f"Combined sequence count: {ml_project.sequence_count()}")
    else:
        ml_project = new_project

    # Upload the annotations as a patch
    patch = create_task_annotations_patch(ml_project, cvat_project.label_to_id_dict)
    cvat_project.create_annotations(task, patch, overwrite=overwrite)


@click.group()
def cli2():
    pass

@cli2.command()
@click.option('--cvat_address', type=str,
              help='Address of CVAT server')
@click.option('--task_id', type=int,
              help='Task ID')
@click.option('--job_id', type=int,
              help='Job ID')
@click.option('--label_to_delete', type=str,
              help='Label that should be deleted from the job')
@click.option('--dry_run/--no-dry_run', default=True,
              help='Only show stats of deleted items, do not delete!')


def delete_annotations(cvat_address, task_id, job_id, label_to_delete, dry_run):
    print(f"Loading task {task_id}")
    url = f"{cvat_address}/api/v1/tasks/{task_id}"
    data = requests.get(url, auth=HTTPBasicAuth('admin', 'admin')).json()
    label_id = None
    for label in data['labels']:
        if  label['name'] == label_to_delete:
            label_id = label['id']
            break
 
    if label_id is None:
        print(f"Error: label {label_to_delete} not found in task {task_id} !")
        return

    print(f"Loading job {job_id}")
    url = f"{cvat_address}/api/v1/jobs/{job_id}/annotations"
    data = requests.get(url, auth=HTTPBasicAuth('admin', 'admin')).json()
    num_tags=0
    for tag in data['tags'].copy():
        if tag['label_id'] == label_id:
            data['tags'].remove(tag)
            num_tags += 1
    num_shapes=0
    for shape in data['shapes'].copy():
        if shape['label_id'] == label_id:
            data['shapes'].remove(shape)
            num_shapes += 1
    num_tracks = 0
    for track in data['tracks'].copy():
        if track['label_id'] == label_id:
            data['tracks'].remove(track)
            num_tracks += 1

    print(f"Deleting {num_tags} tags, {num_shapes} shapes, and {num_tracks} tracks with label {label_to_delete} in job {job_id} (dry_run == {dry_run})!")

    if dry_run:
        return

    url = f"{cvat_address}/api/v1/jobs/{job_id}/annotations?action=create&id={job_id}"
    response = requests.put(url,
                            data=json.dumps(data),
                            auth=HTTPBasicAuth('admin', 'admin'),
                            headers={'Content-Type': "application/json"})
    print(response)


cli = click.CommandCollection(sources=[cli1, cli2])

if __name__ == "__main__":
    cli()
