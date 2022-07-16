import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image

from umami.dataset.parse_cvat_xml import CvatProject, CvatImage, CvatBox


class CvatDataset(torch.utils.data.Dataset):
    def __init__(self, path, transforms=None):
        self.path = path
        self.transforms = transforms

        self.project = CvatProject(os.path.join(self.path, "annotations.xml"))

    def __getitem__(self, idx):
        # Load image
        cvat_image = self.project.images[idx]
        img_path = os.path.join(self.path, "images", cvat_image.name)
        img = Image.open(img_path).convert("RGB")

        # Bounding boxes
        boxes = np.asarray([box.to_array() for box in cvat_image.boxes])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # Labels
        labels = torch.ones((len(boxes),), dtype=torch.int64)

        # Image id
        image_id = torch.tensor([idx])

        # Bounding box areas
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # All instances are not crowd
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.project.images)