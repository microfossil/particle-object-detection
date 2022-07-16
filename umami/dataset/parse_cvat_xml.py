from typing import List

import numpy as np
import xmltodict

class CvatProject():
    def __init__(self, path):
        self.path = path
        self.cls_labels = []
        self.images: List[CvatImage] = []
        self._parse()

    def _parse(self):
        with open(self.path, "r") as fp:
            xml = fp.read()
            obj = xmltodict.parse(xml)

            # Get labels
            for label in obj['annotations']['meta']['task']['labels']['label']:
                self.cls_labels.append(label['name'])

            # Get images
            for image in obj['annotations']['image']:
                ann = CvatImage(image['@name'])
                if "box" in image:
                    for box in image['box']:
                        ann.boxes.append(CvatBox(box['@xtl'], box['@ytl'], box['@xbr'], box['@ybr'], box['@label']))
                    self.images.append(ann)

    def __str__(self):
        print("-" * 80)
        print(f"Cvat Project: {self.path}")
        print("-" * 80)
        print(f"- labels: {self.cls_labels}")
        print(f"- images: {len(self.images)}")
        for image in self.images:
            print()
            print(image.name)
            for box in image.boxes:
                print(f"--- {box.label}: {box.x0} {box.x1} {box.x1} {box.y1}")

class CvatBox():
    def __init__(self, x0, y0, x1, y1, label):
        self.x0 = float(x0)
        self.y0 = float(y0)
        self.x1 = float(x1)
        self.y1 = float(y1)
        self.label = label

    def to_array(self):
        return np.asarray([self.x0, self.y0, self.x1, self.y1])


class CvatImage():
    def __init__(self, name):
        self.name = name
        self.boxes: List[CvatBox] = []

if __name__ == "__main__":
    path = r"C:\Users\ross.marchant\data\task_mixed stacks #1-2022_06_24_09_51_55-cvat for images 1.1\annotations.xml"
    path = path.replace("C:\\", "/mnt/c/")
    path = path.replace("\\", "/")

    project = CvatProject(path)
    print(project)
