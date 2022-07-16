import json
from abc import abstractmethod, ABC, abstractstaticmethod
from collections import OrderedDict
import numpy as np
from lxml import etree


class Annotation(ABC):

    @abstractmethod
    def iou(self, ann):
        pass

    @abstractmethod
    def centre_point_distance(self, ann):
        pass

    @abstractmethod
    def get_bounds(self):
        pass

    @abstractmethod
    def to_cvat(self):
        pass

    @staticmethod
    @abstractmethod
    def from_cvat():
        pass


class RectangleAnnotation(Annotation):
    def __init__(self,
                 x,
                 y,
                 width,
                 height,
                 label,
                 score=1.0,
                 annotator=None,
                 validator=None,
                 uid=None,
                 seq_id=None,
                 seq_idx=0,
                 seq_len=1,
                 frame_id=0,
                 shape="rect",
                 is_keypoint=True):
        """
        Rectangle annotation

        :param x: upper-left x coordinate
        :param y: upper-left y coordinate
        :param width: rectangle width
        :param height: rectangle height
        :param label: class label (string)
        :param score: class score (default 1.0)
        :param annotator: person / model who annotated (string)
        :param validator: person who validated (string)
        :param unique_id: unique id of annotation
        :param seq_id: unique id of object annotation belongs to
        :param seq_idx: index of annotation in sequence of object annotations
        :param seq_len: length of sequence of object annotations
        """
        self.shape = "rect"
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.label = label
        self.score = score
        self.annotator = annotator
        self.validator = validator
        self.uid = uid
        self.seq_id = seq_id
        self.seq_idx = seq_idx
        self.seq_len = seq_len
        self.frame_id = frame_id
        self.is_keypoint = is_keypoint

    def iou(self, ann):
        """
        Intersection-over-union score with another annotation
        0 - no overlap
        1 - perfect overlap
        """
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(self.x, ann.x)
        yA = max(self.y, ann.y)
        xB = min(self.x + self.width, ann.x + ann.width)
        yB = min(self.y + self.height, ann.y + ann.height)
        # compute the area of intersection rectangle
        interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
        if interArea == 0:
            return 0
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = self.width * self.height
        boxBArea = ann.width * ann.height
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou

    def centre_point_distance(self, ann, y_weight=2):
        """
        Get distance between centre point of this annotation and another annotation
        """
        dx = np.abs((self.x + self.width / 2) - (ann.x + ann.width / 2))
        dy = np.abs((self.y + self.height / 2) - (ann.y + ann.height / 2)) / y_weight
        return np.sqrt(dx ** 2 + dy ** 2)

    def get_bounds(self):
        return self.x, self.y, self.width, self.height

    def get_coords(self):
        return self.x, self.y, self.x + self.width, self.y + self.height

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def to_json(self):
        return json.dumps(self.__dict__)

    @classmethod
    def from_json(cls, json_str):
        j = json.loads(json_str, object_pairs_hook=OrderedDict)
        return cls.from_dict(j)

    def to_via(self):
        """
        Convert to VIA format
        """
        annotation = OrderedDict()
        annotation['shape_attributes'] = OrderedDict()
        annotation['shape_attributes']['name'] = "rect"
        annotation['shape_attributes']['x'] = self.x
        annotation['shape_attributes']['y'] = self.y
        annotation['shape_attributes']['width'] = self.width
        annotation['shape_attributes']['height'] = self.height
        annotation['region_attributes'] = OrderedDict()
        annotation['region_attributes']['id'] = self.seq_id
        annotation['region_attributes']['idx'] = self.seq_idx
        annotation['region_attributes']['label'] = self.label
        annotation['region_attributes']['score'] = str(np.round(self.score, 2))
        annotation['region_attributes']['annotator'] = self.annotator
        annotation['region_attributes']['seq_len'] = self.seq_len
        return annotation

    @staticmethod
    def from_via(via_annotation):
        """
        Create from VIA format
        """
        x = int(via_annotation['shape_attributes']['x'])
        y = int(via_annotation['shape_attributes']['y'])
        width = int(via_annotation['shape_attributes']['width'])
        height = int(via_annotation['shape_attributes']['height'])
        if 'id' in via_annotation['region_attributes']:
            id = via_annotation['region_attributes']['id']
            if id == "":
                id = None
        else:
            id = None
        if 'label' in via_annotation['region_attributes']:
            label = via_annotation['region_attributes']['label']
        else:
            label = None
        if 'score' in via_annotation['region_attributes']:
            score = float(via_annotation['region_attributes']['score'])
        else:
            score = 1.0
        if 'annotator' in via_annotation['region_attributes']:
            annotator = via_annotation['region_attributes']['annotator']
        else:
            annotator = "unknown"
        if 'seq_len' in via_annotation['region_attributes']:
            seq_len = via_annotation['region_attributes']['seq_len']
        else:
            seq_len = 1
        if 'idx' in via_annotation['region_attributes']:
            idx = via_annotation['region_attributes']['idx']
        else:
            idx = 0
        return RectangleAnnotation(x,
                                   y,
                                   width,
                                   height,
                                   label,
                                   score=score,
                                   annotator=annotator,
                                   seq_id=id,
                                   seq_idx=idx,
                                   seq_len=seq_len)

    def to_cvat(self):
        attributes = {"label": self.label,
                      "occluded": "0",
                      "source": "manual",
                      "xtl": str(self.x),
                      "ytl": str(self.y),
                      "xbr": str(self.x + self.width),
                      "ybr": str(self.y + self.height),
                      "z_order": "0"}
        el = etree.Element("box", attrib=attributes)
        return el

    def to_cvat_sequenced(self, frame):
        attributes = {"frame": str(frame),
                      "outside": "0",
                      "occluded": "0",
                      "keyframe": "1",
                      "xtl": str(self.x),
                      "ytl": str(self.y),
                      "xbr": str(self.x + self.width),
                      "ybr": str(self.y + self.height),
                      "z_order": "0"}
        el = etree.Element("box", attrib=attributes)
        return el

    @staticmethod
    def from_cvat(el: etree):
        label = el.get("label")
        x = float(el.get("xtl"))
        y = float(el.get("ytl"))
        width = float(el.get("xbr")) - x
        height = float(el.get("ybr")) - y
        return RectangleAnnotation(x, y, width, height, label)


class PolygonAnnotation(Annotation):
    def __init__(self,
                 xs: list,
                 ys: list,
                 label,
                 score=1.0,
                 annotator=None,
                 validator=None,
                 uid=None,
                 seq_id=None,
                 seq_idx=0,
                 seq_len=1,
                 shape="polygon"):
        """
        Rectangle annotation

        :param x: point x coordinates
        :param y: point y coordinates
        :param label: class label (string)
        :param score: class score (default 1.0)
        :param annotator: person / model who annotated (string)
        :param validator: person who validated (string)
        :param unique_id: unique id of annotation
        :param seq_id: unique id of object annotation belongs to
        :param seq_idx: index of annotation in sequence of object annotations
        :param seq_len: length of sequence of object annotations
        """
        self.shape = "polygon"
        self.xs = xs
        self.ys = ys
        self.label = label
        self.score = score
        self.annotator = annotator
        self.validator = validator
        self.uid = uid
        self.seq_id = seq_id
        self.seq_idx = seq_idx
        self.seq_len = seq_len

    def iou(self, ann):
        """
        Intersection-over-union score with another annotation
        0 - no overlap
        1 - perfect overlap
        """
        raise NotImplemented

    def centre_point_distance(self, ann, y_weight=2):
        """
        Get distance between centre point of this annotation and another annotation
        """
        raise NotImplemented

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def to_json(self):
        return json.dumps(self.__dict__)

    @classmethod
    def from_json(cls, json_str):
        j = json.loads(json_str, object_pairs_hook=OrderedDict)
        return cls.from_dict(j)

    def to_via(self):
        """
        Convert to VIA format
        """
        raise NotImplemented

    @staticmethod
    def from_via(via_annotation):
        """
        Create from VIA format
        """
        raise NotImplemented

    def to_cvat(self):
        point_str = ""
        for point in zip(self.xs, self.ys):
            point_str += f"{point[0]},{point[1]};"
        point_str = point_str[:-1]
        attributes = {"label": self.label,
                      "occluded": "0",
                      "source": "manual",
                      "points": point_str,
                      "z_order": "0"}
        el = etree.Element("polygon", attrib=attributes)
        return el

    def to_cvat_sequenced(self, frame):
        point_str = ""
        for point in zip(self.xs, self.ys):
            point_str += f"{point[0]},{point[1]};"
        point_str = point_str[:-1]
        attributes = {"frame": str(frame),
                      "outside": "0",
                      "occluded": "0",
                      "keyframe": "1",
                      "points": point_str,
                      "z_order": "0"}
        el = etree.Element("polygon", attrib=attributes)
        return el

    @staticmethod
    def from_cvat(el: etree):
        label = el.get("label")
        points = el.get("points")
        pairs = points.split(";")
        xs = []
        ys = []
        for pair in pairs:
            parts = pair.split(",")
            x = float(parts[0])
            y = float(parts[1])
            xs.append(x)
            ys.append(y)
        return PolygonAnnotation(xs, ys, label)





if __name__ == "__main__":
    rect = RectangleAnnotation(100, 200, 400, 600, "test", 1.0, "ross", "greg", 1, 101, 2, 45)
    print(rect.to_dict())
    print(rect.to_json())
    print(rect.to_via())
    print(RectangleAnnotation.from_json(rect.to_json()).to_json())
    print(RectangleAnnotation.from_via(rect.to_via()).to_json())
    print(etree.tostring(rect.to_cvat(), pretty_print=True))



    poly = PolygonAnnotation([844.26,871.20,897.63,894.07],
                             [1501.01,1520.83,1513.21,1495.42],
                             "Diseased")
    print(poly.to_dict())
    print(poly.to_json())
    print(PolygonAnnotation.from_json(poly.to_json()).to_json())
    print(etree.tostring(poly.to_cvat(), pretty_print=True))
    el = etree.fromstring(etree.tostring(poly.to_cvat(), pretty_print=True))
    print(etree.tostring(el))
    poly = PolygonAnnotation.from_cvat(el)
    print(poly.to_json())
