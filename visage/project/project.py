import copy
import json
import os
import platform
import re
from collections import OrderedDict
from glob import glob
from io import StringIO
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from lxml import etree
import skimage.io as skio

from visage.project.annotation import RectangleAnnotation, PolygonAnnotation
from visage.project.image import ImageMetadata
from visage.project.default_via_project import get_default_via_project_json


class Project(object):
    def __init__(self) -> object:
        self.filename = ""
        # Images (filename, metadata, annotations)
        self.image_dict: OrderedDict[str, ImageMetadata] = OrderedDict()
        # Project labels (from the image metadata)
        self.label_dict = OrderedDict()
        # Project annotators
        self.user_dict = OrderedDict()
        # Metadata
        self.metadata = OrderedDict()
        # VIA project format (this will be set if loaded from via, it is for carrying over the project settings during save)
        self.default_via = None
        # Add the default metadata
        self.add_default_metadata()

    def format_time(self, time: datetime):
        return time.strftime("%y%m%dT%H%M%S%f")[:-3] + "Z"

    def add_default_metadata(self):
        self.metadata['instrument'] = "unknown"
        time_now = self.format_time(datetime.now())
        self.metadata['created_on'] = time_now
        self.metadata['last_accessed'] = time_now
        self.metadata['last_modified'] = time_now

    def add_image(self, metadata: ImageMetadata, key=None, prefix=None):

        if not key:
            key = metadata.get_key()
            if prefix:
                key = prefix + "_" + key
        # if ensure_unique:
        #     i = 0
        #     new_key = f"{i:04d}_{key}"
        #     # if new_key in self.image_dict:
        #     #     i = 1
        #     #     new_key = f"{i:04d}_{key}"
        #     while new_key in self.image_dict:
        #         i += 1
        #         new_key = f"{i:04d}_{key}"
        #     key = new_key
        #     if i > 20:
        #         raise ValueError("Something is wrong, way too many non-unique image filenames")

        self.image_dict[key] = metadata
        for annotation in metadata.annotations:
            if annotation.label not in self.label_dict:
                self.add_label(annotation.label)
            if annotation.annotator not in self.user_dict:
                self.add_annotator(annotation.annotator)
            if annotation.validator not in self.user_dict:
                self.add_annotator(annotation.validator)

    def image_dict_by_datetime(self):
        dt_image_metadata = OrderedDict()
        for k, v in self.image_dict.items():
            dt = re.search(r"\d{8}_\d{9}", v.get_key()).group()
            dt_image_metadata[dt] = v
        return sorted(dt_image_metadata)

    def update_image_directory(self, new_path):
        for k, v in self.image_dict.items():
            if platform.system() == 'Linux' or platform.system() == "Darwin":
                filename = v.filename.replace("\\", "/")
            else:
                filename = v.filename.replace("/", "\\")
            filename = os.path.basename(filename)
            if new_path.startswith("http"):
                v.filename = new_path + "/" + filename
            else:
                v.filename = os.path.join(new_path, filename)

    def update_image_filename_from_key(self, extension=".jpg"):
        for k, v in self.image_dict.items():
            v.filename = os.path.join(os.path.dirname(v.filename), k + extension)

    def use_image_filename_as_key(self):
        new_dict = OrderedDict()
        for k, v in self.image_dict.items():
            new_dict[v.filename] = v
        self.image_dict = new_dict

    def update_image_directory_start(self, old_path, new_path):
        for k, v in self.image_dict.items():
            if old_path is None or old_path == "":
                fn = v.filename
                if fn.startswith("/") or fn.startswith("\\"):
                    fn = fn[1:]
                v.filename = os.path.join(new_path, fn)
                print(v.filename)
            elif v.filename.startswith(old_path):
                filename = new_path + v.filename[len(old_path):]
                if platform.system() == 'Linux' or platform.system() == "Darwin" or new_path.startswith("http"):
                    filename = filename.replace("\\", "/")
                else:
                    filename = filename.replace("/", "\\")
                v.filename = filename

    def update_label(self, original_label, new_label):
        self.label_dict = OrderedDict(
            [(new_label, v) if k == original_label else (k, v) for k, v in self.label_dict.items()])
        for image_idx, metadata in self.image_dict.items():
            for annotation in metadata.annotations:
                if annotation.label == original_label:
                    annotation.label = new_label
        # if self.default_label is not None and self.default_label == original_label:
        #     self.default_label = new_label

    def add_label(self, label, colour=None):
        if label not in self.label_dict.keys():
            if not colour or colour == "":
                colour = [np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)]
                colour = f"#{colour[0]:02X}{colour[1]:02X}{colour[2]:02X}"
            self.label_dict[label] = colour

    def add_labels(self, labels: list):
        for label in labels:
            self.add_label(label)

    def keep_annotations_with_labels(self, labels):
        for k, metadata in self.image_dict.items():
            new_annotations = []
            for ann in metadata.annotations:
                if ann.label in labels:
                    new_annotations.append(ann)
            metadata.annotations = new_annotations
        old_labels = list(self.label_dict.keys())
        for label in old_labels:
            if label not in labels:
                self.label_dict.pop(label)

    def remove_annotations_with_labels(self, labels):
        for k, metadata in self.image_dict.items():
            new_annotations = []
            for ann in metadata.annotations:
                if ann.label not in labels:
                    new_annotations.append(ann)
            metadata.annotations = new_annotations
        for label in labels:
            self.label_dict.pop(label)

    def remove_annotations_below_threshold(self, threshold):
        for k, metadata in self.image_dict.items():
            new_annotations = []
            for ann in metadata.annotations:
                if ann.score >= threshold:
                    new_annotations.append(ann)
            metadata.annotations = new_annotations

    def add_suffix_to_labels(self, suffix):
        for k, metadata in self.image_dict.items():
            for ann in metadata.annotations:
                ann.label = ann.label + "_" + suffix

    def update_label_dict_from_metadata(self):
        for k, metadata in self.image_dict.items():
            for ann in metadata.annotations:
                if ann.label not in self.label_dict:
                    self.add_label(ann.label)

    def rescale_annotations(self, factor):
        for k, metadata in self.image_dict.items():
            metadata.rescale_annotation(factor)

    def remove_all_annotations(self):
        for k, metadata in self.image_dict.items():
            metadata.annotations = []

    def label_count(self):
        count_dict = {k: 0 for k in self.label_dict.keys()}
        for key, metadata in self.image_dict.items():
            for ann in metadata.annotations:
                if ann.label not in count_dict:
                    count_dict[ann.label] = 1
                    self.label_dict[ann.label] = ""
                else:
                    count_dict[ann.label] += 1
        return count_dict

    def image_with_label_count(self):
        count_dict = {k: 0 for k in self.label_dict.keys()}
        for key, metadata in self.image_dict.items():
            labels = set([ann.label for ann in metadata.annotations])
            for label in labels:
                count_dict[label] += 1
        return count_dict

    def labels_as_json(self):
        label_list = []
        for i, (label, colour) in enumerate(self.label_dict.items()):
            d = {"name": label,
                 "color": colour,
                 "attributes": []}
            label_list.append(d)
        return json.dumps(label_list, indent=4)

    def add_annotator(self, annotator, description=""):
        self.user_dict[annotator] = description

    def add_annotators(self, annotators: list):
        for annotator in annotators:
            self.add_annotator(annotator)

    def add_sensor_data(self, sensor_data_csv: str):
        """
        Add sensor data according to the datetime
        """
        df = pd.read_csv(sensor_data_csv, index_col=0, parse_dates=['datetime'])
        for key, metadata in tqdm(self.image_dict.items()):
            df_key = pd.to_datetime(metadata.datetime, format='%Y%m%d_%H%M%S%f')
            try:
                row = df.loc[df_key]
                # metadata.datetime = df_key.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                metadata.latitude = row['lat']
                if np.abs(metadata.latitude) > 180:
                    metadata.latitude /= 10000000
                metadata.longitude = row['lon']
                if np.abs(metadata.longitude) > 180:
                    metadata.longitude /= 10000000
                metadata.depth = row['fluid_pressure']
                metadata.altitude = row['sonar.distance']
            except KeyError:
                "No sensor data"

    def append_project(self, project, prefix=True):
        for key, metadata in project.image_dict.items():
            self.add_image(metadata, prefix=prefix)

    def extract_datetime_from_filenames(self):
        for key, metadata in self.image_dict.items():
            try:
                dt = re.search(r"\d{8}_\d{9}", metadata.filename).group()
                metadata.metadata["datetime"] = pd.to_datetime(dt, format='%Y%m%d_%H%M%S%f')
            except:
                try:
                    dt = re.search(r"\d{8}T\d{9}", metadata.filename).group()
                    metadata.metadata["datetime"] = pd.to_datetime(dt, format='%Y%m%dT%H%M%S%f')
                except:
                    pass

    def sequence_annotations(self, match_label=False, max_dist=100, time_from_filename=False):
        previous_time = None
        previous_time_difference = None
        previous = None
        next_id = 0
        seq_len_dict = dict()
        ii = 0
        if time_from_filename:
            self.extract_datetime_from_filenames()
        # Remove previous annotation
        for key, metadata in self.image_dict.items():
            for ann in metadata.annotations:
                ann.seq_id = None
                ann.seq_idx = 0
                ann.seq_len = 0
        for key, metadata in tqdm(self.image_dict.items()):
            if previous is None:
                # No ids have been set, so start!
                for i, ann in enumerate(metadata.annotations):
                    ann.seq_id = next_id
                    next_id += 1
                    seq_len_dict[ann.seq_id] = 1
                previous = [copy.deepcopy(ann) for ann in metadata.annotations]
                try:
                    previous_time = metadata.metadata["datetime"]
                except Exception as ex:
                    # print(ex)
                    previous = None
                    continue
            else:
                # Match to previous ids
                # - create a matrix of the distance between each label (row = current, col = previous), weighting for y
                # - select the maximum and assign label if above min iou
                # - remove these indicies (set to zero)
                # - repeat until none left
                # - assign new labels to others
                # - predict position of label for next comparison
                current = [copy.deepcopy(ann) for ann in metadata.annotations]
                current_time = metadata.metadata["datetime"]
                current_time_difference = current_time - previous_time
                if previous_time_difference is None:
                    update_factor = 1
                else:
                    update_factor = current_time_difference / previous_time_difference
                for ann in current:
                    if hasattr(ann, 'dx'):
                        ann.x = ann.x + ann.dx * update_factor
                        ann.y = ann.y + ann.dy * update_factor
                dist = []
                for i, ann in enumerate(current):
                    if not match_label:
                        dist.append(np.asarray([ann.centre_point_distance(b, y_weight=2) for b in previous]))
                    else:
                        dist.append(np.asarray([ann.centre_point_distance(b, y_weight=2) if ann.label == b.label else np.inf for b in previous]))
                dist = np.asarray(dist)
                if np.size(dist) > 0:
                    while True:
                        max_idx = np.unravel_index(np.argmin(dist), dist.shape)
                        max_val = dist[max_idx[0], max_idx[1]]
                        mi = int(max_idx[0])
                        mj = int(max_idx[1])
                        if max_val < max_dist:
                            # Matched, set current id to previous id
                            current[mi].seq_id = previous[mj].seq_id
                            current[mi].seq_idx = previous[mj].seq_idx + 1
                            metadata.annotations[mi].seq_id = previous[mj].seq_id
                            metadata.annotations[mi].seq_idx = previous[mj].seq_idx + 1
                            # Clear row and column belonging to this pair
                            dist[mi, :] = max_dist + 1
                            dist[:, mj] = max_dist + 1
                            # Update the displacement delta
                            current[mi].dx = current[mi].x - previous[mj].x
                            current[mi].dy = current[mi].y - previous[mj].y
                            # Increment the sequence length dictionary for this object
                            seq_len_dict[current[mi].seq_id] += 1
                        else:
                            break
                # Any annotations that were not matched are added
                for i, ann in enumerate(metadata.annotations):
                    if ann.seq_id is None:
                        ann.seq_id = next_id
                        ann.seq_idx = 0
                        seq_len_dict[ann.seq_id] = 1
                        current[i].seq_id = next_id
                        next_id += 1
                previous = current
                previous_time_difference = current_time_difference
        for key, metadata in tqdm(self.image_dict.items()):
            for ann in metadata.annotations:
                ann.seq_len = seq_len_dict[ann.seq_id]


    def remove_sequences_by_length(self, min_length):
        for key, metadata in self.image_dict.items():
            new_annotations = []
            for ann in metadata.annotations:
                if ann.seq_len >= min_length:
                    new_annotations.append(ann)
            metadata.annotations = new_annotations

    def remove_overlapping_annotations(self, master_labels, iou_threshold=0.7):
        for key, im in self.image_dict.items():
            master_anns = [ann for ann in im.annotations if ann.label in master_labels]
            other_anns = [ann for ann in im.annotations if ann.label not in master_labels]
            valid_other_anns = []
            for ann in other_anns:
                is_overlapping = False
                for master_ann in master_anns:
                    iou = ann.iou(master_ann)
                    if iou > iou_threshold:
                        is_overlapping = True
                if not is_overlapping:
                    valid_other_anns.append(ann)
            master_anns.extend(valid_other_anns)
            im.annotations = master_anns

    def add_annotations_from_project(self,
                                     new_project,
                                     add_missing_images=True,
                                     add_missing_annotations=True,
                                     overwrite_annotation_dimensions=False,
                                     overwrite_annotation_label=False,
                                     iou_threshold=0.5,
                                     add_all_annotations=False):
        print("Adding data from {}".format(new_project.filename))
        image_metadata_to_add = []
        count = 0
        overlap_count = 0
        new_image_count = 0
        annotation_count = 0
        new_annotation_count = 0

        ann_dict = dict()
        overlap_dict = dict()
        new_dict = dict()

        missing_key_count = 0

        # TODO when combining, need to ensure unique sequence ids

        # Loop through images in new project
        for key, new_metadata in new_project.image_dict.items():
            # If the image is already in the old project ...
            if key in self.image_dict:
                metadata = self.image_dict[key]
                # Check at each the annotation in the new project image metadata ...
                for new_annotation in new_metadata.annotations:
                    annotation_count += 1
                    if new_annotation.label in ann_dict:
                        ann_dict[new_annotation.label] += 1
                    else:
                        ann_dict[new_annotation.label] = 1
                    overlap = False
                    # ... against each in the current project,
                    for annotation in metadata.annotations:
                        # and if there is any overlap then flag it
                        # print(annotation.iou(new_annotation))
                        if annotation.iou(new_annotation) > iou_threshold:
                            overlap = True
                            overlap_count += 1
                            if overwrite_annotation_dimensions:
                                annotation.x = new_annotation.x
                                annotation.y = new_annotation.y
                                annotation.width = new_annotation.width
                                annotation.height = new_annotation.height
                            if overwrite_annotation_label:
                                annotation.label = new_annotation.label
                                annotation.score = new_annotation.score
                                annotation.annotator = new_annotation.annotator
                            if new_annotation.label in overlap_dict:
                                overlap_dict[new_annotation.label] += 1
                            else:
                                overlap_dict[new_annotation.label] = 1
                            # break
                    # but if no overlap, then add the annotation from the new_project
                    if not add_all_annotations:
                        # print(new_annotation.label)
                        if overlap is False and add_missing_annotations is True:
                            metadata.add_annotation(new_annotation)
                            new_annotation_count += 1
                            if new_annotation.label in new_dict:
                                new_dict[new_annotation.label] += 1
                            else:
                                new_dict[new_annotation.label] = 1
                    else:
                        metadata.add_annotation(new_annotation)
                        new_annotation_count += 1
                        if new_annotation.label in new_dict:
                            new_dict[new_annotation.label] += 1
                        else:
                            new_dict[new_annotation.label] = 1
            # Image not found - add annotation
            elif add_missing_images:
                new_image_count += 1
                image_metadata_to_add.append(new_metadata)
                annotation_count += len(new_metadata.annotations)
                new_annotation_count += len(new_metadata.annotations)
            else:
                missing_key_count += 1
        print("Results")
        print(" - annotations processed: {}, {}".format(annotation_count, ann_dict))
        print(" --- overlapping: {}, {}".format(overlap_count, overlap_dict))
        print(" --- added: {}, {}".format(new_annotation_count - overlap_count, new_dict))
        print(" - images added: {}".format(new_image_count))
        print(" - # keys that could not be matched: {}".format(missing_key_count))
        for metadata in image_metadata_to_add:
            self.add_image(metadata)

    def sequence_dict(self):
        im_idx = 0
        track_dict = OrderedDict()
        for key, im in self.image_dict.items():
            for ann in im.annotations:
                if ann.seq_id not in track_dict:
                    track_dict[ann.seq_id] = []
                ann.frame_id = im_idx
                track_dict[ann.seq_id].append(ann)
            im_idx += 1
        return track_dict

    def sequence_count(self):
        seq = dict()
        already_seen = set()
        for key, metadata in self.image_dict.items():
            for ann in metadata.annotations:
                if ann.label not in seq:
                    seq[ann.label] = []
                if ann.seq_id not in already_seen:
                    seq[ann.label].append(ann.seq_len)
                    already_seen.add(ann.seq_id)
        counts = {key: len(val) for key, val in seq.items()}
        return counts

    def match_sequences(self, project, iou_threshold=0.5, min_overlap=1):
        """
        Returns two projects consisting of the matched and unmatched sequences

        :param project: Project to match with this one
        """
        # TODO
        pass

    def overwrite_project(self, project):
        """
        Overwrites the image metadata in this project with that from another project. Does NOT add new images
        :param project: Project with data to insert
        :return:
        """
        print("Overwriting with data from {}".format(project.filename))
        overwritten_count = 0
        annotations_different = 0
        # Loop through images in new project
        for key, new_metadata in project.image_dict.items():
            # If the image is already in the old project ...
            if key in self.image_dict:
                if len(self.image_dict[key].annotations) != len(new_metadata.annotations):
                    annotations_different += 1
                self.image_dict[key] = new_metadata
                overwritten_count += 1
        print(" - images in this project: {}".format(len(self.image_dict)))
        print(" - images in other project: {}".format(len(project.image_dict)))
        print(" - total overwritten: {}".format(overwritten_count))
        print(" - different no. annotations: {}".format(annotations_different))

    def to_dict(self):
        d = OrderedDict()
        d['metadata'] = self.metadata
        d['labels'] = self.label_dict
        d['users'] = self.user_dict
        d['images'] = OrderedDict()
        for key, val in self.image_dict.items():
            d['images'][key] = val.to_dict()
        return d

    @classmethod
    def from_dict(cls, d):
        project = cls()
        project.metadata = d['metadata']
        project.label_dict = d['labels']
        project.user_dict = d['users']
        for key, val in d['images'].items():
            project.add_image(ImageMetadata.from_dict(val), key=key)
        return project

    def to_json(self, indent=4):
        # print(self.to_dict())
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str):
        return cls.from_dict(json.loads(json_str, object_pairs_hook=OrderedDict))

    def to_via(self):
        # Initialise project dictionary
        if self.default_via is None:
            data = json.loads(get_default_via_project_json(), object_pairs_hook=OrderedDict)
        else:
            data = self.default_via
            if 'id' not in data['_via_attributes']['region']:
                data['_via_attributes']['region']['id'] = OrderedDict()
                data['_via_attributes']['region']['id']['type'] = "text"
                data['_via_attributes']['region']['id']['description'] = ""
                data['_via_attributes']['region']['id']['default_value'] = "null"
            if 'label' not in data['_via_attributes']['region']:
                data['_via_attributes']['region']['label'] = OrderedDict()
                data['_via_attributes']['region']['label']['type'] = "radio"
                data['_via_attributes']['region']['label']['description'] = ""
                data['_via_attributes']['region']['label']['options'] = OrderedDict()
                data['_via_attributes']['region']['label']['default_options'] = OrderedDict()
            if 'score' not in data['_via_attributes']['region']:
                data['_via_attributes']['region']['score'] = OrderedDict()
                data['_via_attributes']['region']['score']['type'] = "text"
                data['_via_attributes']['region']['score']['description'] = ""
                data['_via_attributes']['region']['score']['default_value'] = 1.0
            if 'annotator' not in data['_via_attributes']['region']:
                data['_via_attributes']['region']['annotator'] = OrderedDict()
                data['_via_attributes']['region']['annotator']['type'] = "text"
                data['_via_attributes']['region']['annotator']['description'] = ""
                data['_via_attributes']['region']['annotator']['default_value'] = "unknown"
            if 'seq_len' not in data['_via_attributes']['region']:
                data['_via_attributes']['region']['seq_len'] = OrderedDict()
                data['_via_attributes']['region']['seq_len']['type'] = "text"
                data['_via_attributes']['region']['seq_len']['description'] = ""
                data['_via_attributes']['region']['seq_len']['default_value'] = 1
            for key in ['datetime', 'latitude', 'longitude', 'depth', 'altitude', 'tagged']:
                if key not in data['_via_attributes']['file']:
                    data['_via_attributes']['file'][key] = OrderedDict()
                    data['_via_attributes']['file'][key]['type'] = "text"
                    data['_via_attributes']['file'][key]['description'] = ""
                    data['_via_attributes']['file'][key]['default_value'] = ""
        # Clear previous data
        data['_via_img_metadata'] = OrderedDict()
        data['_via_attributes']['region']['label']['options'] = OrderedDict()
        # data['_via_attributes']['region']['label']['default_options'] = OrderedDict()
        data['_via_attributes']['region']['annotator']['options'] = OrderedDict()
        # data['_via_attributes']['region']['annotator']['default_options'] = OrderedDict()
        # Add all images
        for image_key, metadata in self.image_dict.items():
            data['_via_img_metadata'][metadata.filename + str(metadata.filesize)] = metadata.to_via()
        # Add all labels
        for label_key, description in self.label_dict.items():
            data['_via_attributes']['region']['label']['options'][label_key] = description
        # Add all annotators
        for ann_key, description in self.user_dict.items():
            data['_via_attributes']['region']['annotator']['options'][ann_key] = description
        # Set image directory if not blank (e.g. when using URLs)
        if 'default_filepath' not in data['_via_settings']['core']:
            data['_via_settings']['core']['default_filepath'] = ""
        # Set project filename (without extension)
        if 'name' not in data['_via_settings']['project']:
            data['_via_settings']['project']['name'] = os.path.basename(self.filename)[:-5]
        return data

    @staticmethod
    def from_via(data):
        project = Project()
        # Add all the labels from the project
        if 'label' in data['_via_attributes']['region']:
            # Add the the labels
            for key, description in data['_via_attributes']['region']['label']['options'].items():
                project.label_dict[key] = description
        # Add all the file attributes from the project
        if 'annotator' in data['_via_attributes']['region']:
            # Add the the annotators
            for key, description in data['_via_attributes']['region']['annotator']['options'].items():
                project.user_dict[key] = description
        # Add all the images. This will add any missing labels as well
        for key, metadata in data['_via_img_metadata'].items():
            image_metadata = ImageMetadata.from_via(metadata)
            project.add_image(image_metadata)
        return project

    def to_cvat(self, default_image_size=(2048, 2448)):
        root = etree.Element("annotations")
        version = etree.SubElement(root, "version")
        version.text = "1.1"

        # labels_elem = etree.SubElement(root, "labels")
        # for i, (label, colour) in enumerate(self.label_dict.items()):
        #     label_elem = etree.SubElement(labels_elem, "label")
        #     name = etree.SubElement(label_elem, "name")
        #     name.text = label
        #     clr = etree.SubElement(label_elem, "color")
        #     clr.text = colour
        #     attr = etree.SubElement(label_elem, "attributes")

        im_idx = 0
        for key, im in self.image_dict.items():
            im.metadata["id"] = im_idx
            im_idx += 1
            if "height" not in im.metadata:
                im.metadata["height"] = default_image_size[0]
            if "width" not in im.metadata:
                im.metadata["width"] = default_image_size[1]
            im_elem = im.to_cvat()
            for ann in im.annotations:
                im_elem.append(ann.to_cvat())
            root.append(im_elem)
        return root

    def to_cvat_sequenced(self, default_image_size=(2048, 2448)):
        root = etree.Element("annotations")
        version = etree.SubElement(root, "version")
        version.text = "1.1"

        # labels_elem = etree.SubElement(root, "labels")
        # for i, (label, colour) in enumerate(self.label_dict.items()):
        #     label_elem = etree.SubElement(labels_elem, "label")
        #     name = etree.SubElement(label_elem, "name")
        #     name.text = label
        #     clr = etree.SubElement(label_elem, "color")
        #     clr.text = colour
        #     attr = etree.SubElement(label_elem, "attributes")

        im_idx = 0
        track_dict = OrderedDict()
        for key, im in self.image_dict.items():
            for ann in im.annotations:
                if ann.seq_id not in track_dict:
                    track_dict[ann.seq_id] = []
                ann.frame_id = im_idx
                track_dict[ann.seq_id].append(ann)
            im_idx += 1
        track_idx = 0
        for key, anns in track_dict.items():
            attr = {"id": str(track_idx), "label": anns[0].label, "source": "manual"}
            elem = etree.Element("track", attr)
            for ann in anns:
                elem.append(ann.to_cvat_sequenced(ann.frame_id))
            final = copy.deepcopy(ann.to_cvat_sequenced(ann.frame_id + 1))
            final.set("outside", "1")
            elem.append(final)
            root.append(elem)
            track_idx += 1
        return root

    @staticmethod
    def from_cvat(root: etree):
        project = Project()
        # Metadata
        project.metadata["cvat_project_id"] = root.find(".//meta/task/id")
        project.metadata["cvat_project_id"] = root.find(".//meta/task/id")
        project.metadata["cvat_project_id"] = root.find(".//meta/task/id")
        # Labels
        for label_elem in root.findall(".//label"):
            project.add_label(label_elem.find("name").text, label_elem.find("color").text)
        # Images
        image_elements = root.findall(".//image")
        for image_element in image_elements:
            im = ImageMetadata.from_cvat(image_element)
            if "mpv-shot" not in im.filename:  # Hack if someone has accidentally saved an mpv shot in there
                for element in image_element:
                    if element.tag == 'polygon':
                        ann = PolygonAnnotation.from_cvat(element)
                        im.add_annotation(ann)
                    if element.tag == 'box':
                        ann = RectangleAnnotation.from_cvat(element)
                        im.add_annotation(ann)
                project.add_image(im)
        return project

    def to_mlboxv2(self):
        idx = 0
        with StringIO() as s:
            for key, metadata in self.image_dict.items():
                s.write(f"{metadata.filename}, ")
                for ann in metadata.annotations:
                    if ann.seq_id is None:
                        seq_id = idx
                        idx += 1
                    else:
                        seq_id = ann.seq_id
                    s.write("{" + f"{ann.label}, {ann.score}, {seq_id}, {ann.seq_idx},"
                                  f" {ann.x}, {ann.y}, {ann.width}, {ann.height}" + "}, ")
                s.write("\n")
            return s.getvalue()

    # def to_pandas(self):
    #     data = OrderedDict()
    #     data['#glider_name'] = []
    #     data['#run'] = []
    #     data['#timestamp'] = []
    #     data['image'] = []
    #     data['class'] = []
    #     data['cx'] = []
    #     data['cy'] = []
    #     data['width'] = []
    #     data['height'] = []
    #     data['seq_id'] = []
    #     data['seq_len'] = []
    #     data['annotator'] = []
    #     for key, metadata in self.image_dict.items():
    #         for ann in metadata.annotations:
    #             data['#glider_name'].append(self.glider_name)
    #             data['#run'].append(self.run_name)
    #             ts = datetime.datetime.strptime(key, '%Y%m%d_%H%M%S%f')
    #             data['#timestamp'].append(ts)
    #             data['image'].append(metadata.filename)
    #             data['seq_id'].append(ann.seq_id)
    #             data['seq_len'].append(ann.seq_len)
    #             data['class'].append(ann.label)
    #             data['cx'].append(ann.y + ann.width // 2)
    #             data['cy'].append(ann.y + ann.height // 2)
    #             data['width'].append(ann.width)
    #             data['height'].append(ann.height)
    #             data['annotator'].append(ann.annotator)
    #     return pd.DataFrame(data)

    def summary(self):
        print('-' * 50)
        print("Project: {}".format(self.filename))
        print('-' * 50)
        print("Labels:")
        for key, val in self.label_dict.items():
            print(" - {}".format(key))
        print("Annotators:")
        for key, val in self.user_dict.items():
            print(" - {}".format(key))
        print("Stats:")
        print("- number of images: {}".format(len(self.image_dict)))
        print("- sequences: {}".format(self.sequence_count()))

    @staticmethod
    def load(filename):
        if filename.lower().endswith(".xml"):
            root = etree.parse(filename)
            project = Project.from_cvat(root)
        # elif filename.lower().endswith(".csv") or filename.lower().endswith(".bbx"):
        #     project = convert_mlboxv2_as_is(filename, remove_from_start="")
        else:
            with open(filename, 'r') as f:
                data = json.load(f, object_pairs_hook=OrderedDict)
            if "_via_img_metadata" in data:
                project = Project.from_via(data)
                project.default_via = data
            else:
                project = Project.from_dict(data)
        project.metadata['last_accessed'] = project.format_time(datetime.now())
        project.extract_datetime_from_filenames()
        project.filename = filename
        return project

    @staticmethod
    def from_directory(path):
        filenames = sorted(glob(os.path.join(path, "*.jpg")))
        project = Project()
        for filename in filenames:
            # print(filename)
            project.add_image(ImageMetadata(filename))
        return project

    def save_as(self, filename, format=None, indent=None):
        self.filename = filename
        self.metadata['last_modified'] = self.format_time(datetime.now())
        if os.path.dirname(filename) != "":
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            if format is None:
                f.write(self.to_json(indent=indent))
            elif format == "via":
                f.write(json.dumps(self.to_via(), indent=indent))
            elif format == "cvat":
                f.write(etree.tostring(self.to_cvat(), pretty_print=True).decode("UTF-8"))
            elif format == "cvat_seq":
                f.write(etree.tostring(self.to_cvat_sequenced(), pretty_print=True).decode("UTF-8"))
            elif format == "mlboxv2":
                f.write(self.to_mlboxv2())

    def copy(self):
        return Project.from_dict(self.to_dict())

    def export_patches(self, output_dir, name, valid_labels=None, buffer=100):
        base_dir = os.path.join(output_dir, name)
        os.makedirs(base_dir, exist_ok=True)
        for key, im in tqdm(project.image_dict.items()):
            if len(im.annotations) > 0:
                # Make sure this image has annotations we are interested in
                if valid_labels is not None:
                    if not any(x in im.annotations for x in valid_labels):
                        continue
                img = skio.imread(im.filename)
                for ann in im.annotations:
                    # Skip if not an annotation we are interested in
                    if valid_labels is not None and ann.label not in valid_labels:
                        continue
                    path = os.path.join(output_dir, ann.label, str(ann.seq_id))
                    os.makedirs(path, exist_ok=True)
                    x = int(ann.x) - buffer
                    y = int(ann.y) - buffer
                    w = int(ann.width) + buffer * 2
                    h = int(ann.height) + buffer * 2
                    sub_img = img[y:y + h, x:x + w, ...]
                    skio.imsave(os.path.join(path, f"{ann.seq_idx}.jpg"), sub_img, check_contrast=False)



