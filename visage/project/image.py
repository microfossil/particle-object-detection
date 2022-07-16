import json
import os
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import List

from lxml import etree
from pandas import Timestamp

from visage.project.annotation import RectangleAnnotation


class ImageMetadata(object):
    def __init__(self,
                 filename,
                 filesize=0,
                 metadata: dict = None):
        self.filename = filename
        self.filesize = filesize
        if metadata is None:
            metadata = OrderedDict()
        self.metadata = metadata
        self.annotations: List[RectangleAnnotation] = []

    def get_key(self):
        return str(Path(self.filename).stem)

    def add_annotation(self, annotation: RectangleAnnotation):
        self.annotations.append(annotation)

    def rescale_annotation(self, factor):
        for ann in self.annotations:
            ann.x = ann.x * factor
            ann.y = ann.y * factor
            ann.width = ann.width * factor
            ann.height = ann.height * factor

    def has_label(self, label):
        for ann in self.annotations:
            if ann.label == label:
                return True
        return False

    def has_labels(self, labels: list):
        for label in labels:
            if self.has_label(label):
                return True
        return False

    def labels(self):
        labels = []
        for ann in self.annotations:
            labels.append(ann.label)
        label_set = set(labels)
        return list(label_set)

    def time_from_filename(self, mode=0):
        fn = os.path.basename(self.filename)
        if mode == 0:
            fstr = '%Y%m%dT%H%M%S%fZ'
            dt = fn[-len(fstr)-4:-4]
            return datetime.strptime(dt, '%Y%m%dT%H%M%S%fZ')
        if mode == 1:
            fstr = '%Y%m%d_%H%M%S%f'
            dt = fn[-12-18:-12]
            return datetime.strptime(dt, fstr)

    def to_dict(self):
        d = OrderedDict()
        d['filename'] = self.filename
        d['filesize'] = self.filesize
        d['metadata'] = OrderedDict()
        if self.metadata is not None:
            d['metadata'] = self.metadata
            for k, v in d['metadata'].items():
                if isinstance(v, Timestamp):
                    d['metadata'][k] = None
        d['annotations'] = []
        for ann in self.annotations:
            d['annotations'].append(ann.to_dict())
        return d

    @classmethod
    def from_dict(cls, d):
        c = cls(d['filename'], d['filesize'], d['metadata'])
        for ann in d['annotations']:
            c.add_annotation(RectangleAnnotation.from_dict(ann))
        return c

    def to_json(self):
        return json.dumps(self.to_dict(), indent=4)

    @classmethod
    def from_json(cls, json_str):
        j = json.loads(json_str, object_pairs_hook=OrderedDict)
        return cls.from_dict(j)

    def to_via(self):
        """
        Convert to VIA format
        """
        metadata = OrderedDict()
        metadata['filename'] = self.filename
        metadata['size'] = self.filesize
        metadata['regions'] = list()
        # metadata['file_attributes'] = self.metadata
        for annotation in self.annotations:
            metadata['regions'].append(annotation.to_via())
        return metadata

    @classmethod
    def from_via(cls, metadata):
        image_metadata = cls(metadata['filename'],
                             metadata['size'])
                             # metadata['file_attributes'])
        for region in metadata['regions']:
            image_metadata.add_annotation(RectangleAnnotation.from_via(region))
        return image_metadata

    def to_cvat(self):
        attributes = {"id": str(self.metadata["id"]),
                      "name": self.filename,
                      "width": str(self.metadata["width"]),
                      "height": str(self.metadata["height"])}
        el = etree.Element("image", attrib=attributes)
        return el

    @staticmethod
    def from_cvat(el: etree):
        filename = el.get("name")
        im = ImageMetadata(filename, 0)
        im.metadata["id"] = el.get("id")
        im.metadata["width"] = el.get("width")
        im.metadata["height"] = el.get("height")
        return im


if __name__ == "__main__":
    image_metadata = ImageMetadata("test_filename.jpg", 0)
    rect1 = RectangleAnnotation(100, 200, 400, 600, "test", 1.0, "ross", "greg", 1, 101, 2, 45)
    rect2 = RectangleAnnotation(100, 200, 400, 600, "test", 1.0, "ross", "greg", 1, 101, 2, 45)
    image_metadata.add_annotation(rect1)
    image_metadata.add_annotation(rect2)
    image_metadata.metadata['lat'] = 19.0123
    image_metadata.metadata['lon'] = 145.01234
    print(ImageMetadata.from_dict(image_metadata.to_dict()).to_json())
    print(ImageMetadata.from_json(image_metadata.to_json()).to_json())
    print(json.dumps(image_metadata.to_via(), indent=4))
