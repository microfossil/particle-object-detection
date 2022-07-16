import os
import io
import shutil
import zipfile
from collections import OrderedDict
import json
from pathlib import Path
from time import sleep
from typing import List
import numpy as np
import requests
from requests.auth import HTTPBasicAuth
from visage.project.annotation import RectangleAnnotation
from visage.project.image import ImageMetadata
from visage.project.project import Project


# ----------------------------------------------------------------------------
# Python objects that mirror the JSON objects in the CVAT REST API
#
# 'minimal' function is used to create an instance with the minimum number of
# fields needed for PATCH and PUT.
#
# Fields with None will be removed when serialised
# ----------------------------------------------------------------------------
class CVATJsonSerializable:
    @staticmethod
    def del_none(d):
        """
        Delete keys with the value ``None`` in a dictionary, recursively.

        This alters the input so you may wish to ``copy`` the dict first.
        """
        for key, value in list(d.items()):
            if value is None:
                del d[key]
            elif isinstance(value, dict):
                CVATJsonSerializable.del_none(value)
        return d  # For convenience

    def to_json(self):
        return json.dumps(self, default=lambda o: CVATJsonSerializable.del_none(o.__dict__), sort_keys=True, indent=4)


class CVATTrackedShape(CVATJsonSerializable):
    def __init__(self):
        super(CVATTrackedShape, self).__init__()
        self.type: str = None
        self.occluded: bool = None
        self.z_order: int = None
        self.points: List[int] = []
        self.id: int = None
        self.frame: int = None
        self.outside: bool = None
        self.attributes = []

    @staticmethod
    def minimal(type, occluded, points, frame, outside):
        obj = CVATTrackedShape()
        obj.type = type
        obj.occluded = occluded
        obj.points = points
        obj.frame = frame
        obj.outside = outside
        return obj


class CVATLabeledTrack(CVATJsonSerializable):
    def __init__(self):
        super(CVATLabeledTrack, self).__init__()
        self.id: int = None
        self.frame: int = None
        self.label_id: int = None
        self.group: int = None
        self.source: str = None
        self.shapes: List[CVATTrackedShape] = []
        self.attributes = []

    @staticmethod
    def minimal(frame, label_id, group, shapes):
        obj = CVATLabeledTrack()
        obj.frame = frame
        obj.label_id = label_id
        obj.group = group
        obj.shapes = shapes
        return obj


class CVATLabeledData(CVATJsonSerializable):
    def __init__(self):
        super(CVATLabeledData, self).__init__()
        self.version: int = None
        self.tags = []  # To add when needed
        self.shapes = []  # To add when needed
        self.tracks: List[CVATLabeledTrack] = []

    @staticmethod
    def minimal(version, tracks):
        obj = CVATLabeledData()
        obj.version = version
        obj.tracks = tracks
        return obj


class CVATJob(object):
    def __init__(self, id, task_id, start_frame, stop_frame):
        self.id = id
        self.task_id = task_id
        self.start_frame = start_frame
        self.stop_frame = stop_frame


class CVATTask(object):
    def __init__(self, server: str, task_id: int, task_name: str, debug=True):
        self.server = server
        self.task_id = task_id
        self.task_name = task_name
        self.labels = dict()
        self.debug = debug
        self.frames = []
        self.tracks = dict()
        self.project = Project()

    def load(self):
        if self.debug:
            print("-" * 80)
            print(f"Loading CVAT task {self.task_id} - {self.task_name}")
        self._get_labels()
        self._get_frames()
        self._get_annotations()
        self._get_jobs()
        self._create_project()

    def _create_project(self):
        # Creates a project from this task with interpolation between the key frames
        self.project.metadata["name"] = self.task_name
        self.project.metadata["id"] = self.task_id
        if self.debug:
            print("Creating visage-ml project object...")
        self.project.add_labels(list(self.labels.values()))
        for frame in self.frames:
            self.project.add_image(ImageMetadata(frame))
        for track in self.tracks:
            seq_id = track['id']
            seq_len = len(track['shapes']) - 1
            seq_idx = 0
            label = self.labels[track['label_id']]
            last_frame_idx = None
            last_p = None
            for i, shape in enumerate(track['shapes']):
                frame_idx = shape['frame']
                if shape['type'] == 'rectangle' and shape['outside'] is False:
                    p = np.asarray(shape['points'])
                    # Interpolate
                    if last_frame_idx is not None:
                        if frame_idx - last_frame_idx > 1:
                            for idx in range(last_frame_idx + 1, frame_idx):
                                step = (idx - last_frame_idx) / (frame_idx - last_frame_idx)
                                proj_p = last_p + (p - last_p) * step
                                metadata = self.project.image_dict[str(Path(self.frames[idx]).stem)]
                                metadata.add_annotation(RectangleAnnotation(x=proj_p[0],
                                                                            y=proj_p[1],
                                                                            width=proj_p[2] - proj_p[0],
                                                                            height=proj_p[3] - proj_p[1],
                                                                            label=label,
                                                                            seq_id=seq_id,
                                                                            seq_len=seq_len,
                                                                            seq_idx=seq_idx,
                                                                            frame_id=idx))
                                seq_idx += 1
                    # Current frame
                    metadata = self.project.image_dict[str(Path(self.frames[shape['frame']]).stem)]
                    metadata.add_annotation(RectangleAnnotation(x=p[0],
                                                                y=p[1],
                                                                width=p[2] - p[0],
                                                                height=p[3] - p[1],
                                                                label=label,
                                                                seq_id=seq_id,
                                                                seq_len=seq_len,
                                                                seq_idx=seq_idx,
                                                                frame_id=frame_idx))
                    last_frame_idx = frame_idx
                    last_p = p
                    seq_idx += 1

                elif shape['type'] == 'polygon' and shape['outside'] is False:
                    pass

        for shape in self.shapes:
            seq_id = shape['id']
            seq_len = 1
            seq_idx = 0
            label = self.labels[shape['label_id']]
            frame_idx = shape['frame']
            p = np.asarray(shape['points'])
            if shape['type'] == 'rectangle':
                metadata = self.project.image_dict[str(Path(self.frames[shape['frame']]).stem)]
                metadata.add_annotation(RectangleAnnotation(x=p[0],
                                                            y=p[1],
                                                            width=p[2] - p[0],
                                                            height=p[3] - p[1],
                                                            label=label,
                                                            seq_id=seq_id,
                                                            seq_len=seq_len,
                                                            seq_idx=seq_idx))
                seq_idx += 1

            elif shape['type'] == 'polygon' and shape['outside'] is False:
                pass

        if self.debug:
            print("- label count:")
            for k, v in self.project.label_count().items():
                print(f"   {k}: {v}")
            print("- sequence count:")
            for k, v in self.project.sequence_count().items():
                print(f"   {k}: {v}")

    def _get_frames(self):
        url = f"{self.server}/api/v1/tasks/{self.task_id}/data/meta"
        if self.debug:
            print(f"Fetching task frames from {url}...")
        data = requests.get(url, auth=HTTPBasicAuth('admin', 'admin')).json()
        frames = [frame['name'] for frame in data["frames"]]
        self.frames = frames

    def _get_labels(self):
        url = f"{self.server}/api/v1/tasks/{self.task_id}"
        if self.debug:
            print(f"Fetching task labels from {url}...")
        data = requests.get(url, auth=HTTPBasicAuth('admin', 'admin')).json()
        labels = {label['id']: label['name'] for label in data['labels']}
        self.labels = labels

    def _get_annotations(self):
        url = f"{self.server}/api/v1/tasks/{self.task_id}/annotations"
        if self.debug:
            print(f"Fetching task tracks from {url}...")
        data = requests.get(url, auth=HTTPBasicAuth('admin', 'admin')).json()
        self.tracks = data['tracks']
        self.shapes = data['shapes']
        if self.debug:
            print(f" - {len(self.tracks)} tracks found")

    def _get_jobs(self):
        url = f"{self.server}/api/v1/tasks/{self.task_id}/jobs"
        if self.debug:
            print(f"Fetching jobs from {url}...")
        data = requests.get(url, auth=HTTPBasicAuth('admin', 'admin')).json()
        jobs = {job['id']: CVATJob(job['id'], job['task_id'], job['start_frame'], job['stop_frame']) for job in data}
        self.jobs = jobs


class CVATProject(object):
    def __init__(self, server, project_id, convert_half_size=False, debug=True):
        self.server = server
        self.project_id = project_id
        self.debug = debug

        self.id_to_label_dict = None
        self.label_to_id_dict = None
        self.tasks = dict()
        self.task_to_id_dict = dict()
        self.id_to_task_dict = dict()

    def load(self):
        if self.debug:
            print("=" * 80)
            print(f"Loading CVAT project {self.project_id}")
        self._get_metadata()

    def load_task(self, id):
        if id in self.tasks:
            return self.tasks[id]
        else:
            name = self.id_to_task_dict[id]
            task = CVATTask(self.server, id, name, debug=True)
            task.load()
            # if self.convert_half_size:
            #     self._convert_size(task)
            self.tasks[id] = task
            return task

    def load_task_by_name(self, name):
        id = self.task_to_id_dict[name]
        return self.load_task(id)

    def create_task(self, task_name, filenames):
        url = f"{self.server}/api/v1/tasks"
        if self.debug:
            print(f"Creating new task {task_name}...")
        content = {"project_id": self.project_id, "name": task_name}
        response = requests.post(url, json=content, auth=HTTPBasicAuth('admin', 'admin'))
        print(response)
        id = response.json()["id"]

        url = f"{self.server}/api/v1/tasks/{id}/data"
        if self.debug:
            print(f"Creating files for task {task_name}...")
        content = {"chunk_size": 4,
                   "image_quality": 70,
                   "client_files": [],
                   "server_files": filenames,
                   "remote_files": [],
                   "use_zip_chunks": False,
                   "use_cache": True}
        response = requests.post(url, json=content, auth=HTTPBasicAuth('admin', 'admin'))
        print(response.status_code)
        # print(response.json())
        # Sleep because CVAT sucks
        print("Sleeping to wait for images to be added to project")
        sleep(15)
        self._get_metadata()

    def create_annotations(self, task_name, annotations: CVATLabeledData, overwrite=True):
        url = f"{self.server}/api/v1/tasks/{self.task_to_id_dict[task_name]}/annotations?action=create"
        if self.debug:
            print(f"Creating annotations for task {task_name}...")
        content = annotations.to_json()
        with open("/tmp/content.json", "w") as file:
            file.write(content)
        if overwrite:
            response = requests.put(url,
                                    data=content,
                                    auth=HTTPBasicAuth('admin', 'admin'),
                                    headers={'Content-Type': "application/json"})
        else:
            response = requests.patch(url,
                                      data=content,
                                      auth=HTTPBasicAuth('admin', 'admin'),
                                      headers={'Content-Type': "application/json"})
        print(response)

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Hack because some CVAT projects use the 'jpg_web' directory, which is half-size images
    # We replace the filename with the 'jpg' directory which has the full-size images
    # 2021-01-27 - removed as it requires extra work to undo the process to fix annotations before upload
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # def _convert_size(self, task):
    #     for key, im in task.project.image_dict.items():
    #         if 'jpg_web' in im.filename:
    #             im.filename = im.filename.replace('jpg_web', 'jpg')
    #             for ann in im.annotations:
    #                 ann.x *= 2
    #                 ann.y *= 2
    #                 ann.width *= 2
    #                 ann.height *= 2

    def _get_metadata(self):
        url = f"{self.server}/api/v1/projects/{self.project_id}"
        if self.debug:
            print(f"Fetching project metadata from {url}...")
        data = requests.get(url, auth=HTTPBasicAuth('admin', 'admin')).json()
        self.id_to_label_dict = {label['id']: label['name'] for label in data["labels"]}
        self.label_to_id_dict = {label['name']: label['id'] for label in data["labels"]}
        self.id_to_task_dict = {task['id']: task['name'] for task in data['tasks']}
        self.task_to_id_dict = {task['name']: task['id'] for task in data['tasks']}
        if self.debug:
            print("Tasks:")
            for key, val in self.id_to_task_dict.items():
                print(f" - {key:3d}: {val}")

    def export(self, output_dir, remote_path, local_path, cloud_path):
        if self.debug:
            print('-' * 80)
            print(f'Exporting tasks as project files to {output_dir}')
        for task in self.tasks:
            if self.debug:
                print(f" - {task.task_name}")
            path = os.path.join(output_dir, task.task_name)
            os.makedirs(path, exist_ok=True)
            project = task.project
            project.update_image_directory_start(remote_path, local_path)
            project.save_as(os.path.join(path, f"{task.task_name}_local.json"))
            project.update_image_directory_start(local_path, cloud_path)
            project.save_as(os.path.join(path, f"{task.task_name}_cloud_via.json"), format='via')


class CVAT(object):
    def __init__(self, server, debug=True):
        super(CVAT, self).__init__()
        self.server = server
        self.debug = debug
        self.id_to_project_dict = None
        self.project_to_id_dict = None
        self.projects = dict()

    def load(self):
        if self.debug:
            print("=" * 80)
            print(f"Loading CVAT project list")
        self._get_metadata()

    def load_project(self, id):
        if id in self.projects:
            return self.projects[id]
        else:
            project = CVATProject(self.server, id, convert_half_size=True, debug=True)
            project.load()
            self.projects[id] = project
            return project

    def load_project_by_name(self, name):
        id = self.project_to_id_dict[name]
        return self.load_project(id)

    def load_task(self, project_id, task_id):
        project = self.load_project(project_id)
        task = project.load_task(task_id)
        return task

    def load_task_by_name(self, project_name, task_name):
        project = self.load_project_by_name(project_name)
        task = project.load_task_by_name(task_name)
        return task

    def load_task_by_code(self, code):
        parts = code.split("@")
        project_name = parts[1]
        task_name = parts[0]
        task = self.load_task_by_name(project_name, task_name)
        return task

    def _get_metadata(self):
        url = f"{self.server}/api/v1/projects?names_only=true"
        if self.debug:
            print(f"Fetching projects from {url}...")
        data = requests.get(url, auth=HTTPBasicAuth('admin', 'admin')).json()
        self.id_to_project_dict = {project['id']: project['name'] for project in data["results"]}
        self.project_to_id_dict = {project['name']: project['id'] for project in data["results"]}
        if self.debug:
            print("Projects:")
            for key, val in self.id_to_project_dict.items():
                print(f" - {key:3d}: {val}")





def create_task_annotations_patch(project: Project, label_to_id_dict: dict):
    track_dict = project.sequence_dict()
    cvat_tracks = []
    for id, track in track_dict.items():
        if len(track) > 0:
            cvat_track = CVATLabeledTrack.minimal(frame=track[0].frame_id,
                                                  label_id=label_to_id_dict[track[0].label],
                                                  group=0,
                                                  shapes=[])
            # Actual shapes
            for ann in track:
                cvat_tracked_shape = CVATTrackedShape.minimal(type="rectangle",
                                                              occluded=False,
                                                              points=[ann.x, ann.y, ann.x + ann.width,
                                                                      ann.y + ann.height],
                                                              frame=ann.frame_id,
                                                              outside=False)
                cvat_track.shapes.append(cvat_tracked_shape)
            # Dummy closing shape
            cvat_tracked_shape = CVATTrackedShape.minimal(type="rectangle",
                                                          occluded=False,
                                                          points=[ann.x, ann.y, ann.x + ann.width, ann.y + ann.height],
                                                          frame=ann.frame_id + 1,
                                                          outside=True)
            cvat_track.shapes.append(cvat_tracked_shape)
            cvat_tracks.append(cvat_track)
    return CVATLabeledData.minimal(0, cvat_tracks)


if __name__ == "__main__":
    cvat = CVAT("http://cruncher-ph.nexus.csiro.au:8080")
    cvat.load()

    tasks_list = ["20200717_lamont_1_v4@Heron Island July 2020",
                  "20200716_fitzroy_ff_v4@Heron Island July 2020"]

    for task in tasks_list:
        task = cvat.load_task_by_code(task)
        project = task.project

        project.summary()


# def create_task_annotations_patch(project: Project, label_to_id_dict: dict):
#     track_dict = project.sequence_dict()
#     cvat_tracks = []
#     for id, track in track_dict.items():
#         if len(track) > 0:
#             cvat_track = CVATLabeledTrack.minimal(frame=track[0].frame_id,
#                                                   label_id=label_to_id_dict[track[0].label],
#                                                   group=0,
#                                                   shapes=[])
#             # Actual shapes
#             for ann in track:
#                 cvat_tracked_shape = CVATTrackedShape.minimal(type="rectangle",
#                                                               occluded=False,
#                                                               points=[ann.x, ann.y, ann.x + ann.width,
#                                                                       ann.y + ann.height],
#                                                               frame=ann.frame_id,
#                                                               outside=False)
#                 cvat_track.shapes.append(cvat_tracked_shape)
#             # Dummy closing shape
#             cvat_tracked_shape = CVATTrackedShape.minimal(type="rectangle",
#                                                           occluded=False,
#                                                           points=[ann.x, ann.y, ann.x + ann.width, ann.y + ann.height],
#                                                           frame=ann.frame_id + 1,
#                                                           outside=True)
#             cvat_track.shapes.append(cvat_tracked_shape)
#             cvat_tracks.append(cvat_track)
#     return CVATLabeledData.minimal(0, cvat_tracks)