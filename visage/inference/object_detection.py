import time
from collections import OrderedDict
import tensorflow as tf
from glob import glob
import os
import skimage.io as skio
import numpy as np
from tqdm import tqdm
from pathlib import Path

from visage.inference.coco_metrics import coco_evaluate
from visage.inference.f_beta_metrics import f_beta_score_iou_sweep, f_beta_score
from visage.project.annotation import RectangleAnnotation
from visage.project.image import ImageMetadata
from visage.project.project import Project
from visage.tracking.new_tracker import Tracker, TrackAnnotation


# TODO convert to using the rectangle annotation class


class ObjectDetector(object):
    def __init__(self, model_path, cls_labels=None, gpu=0):
        """
        Object detection for models trained with Tensorflow Object Detection API

        :param model_path: path to the saved_model directory (TF2) or frozen graph (.pb file TF1)
        :param cls_labels: list of labels, the first is always the background class (not a class used in training). E.g. ['background', 'class1', 'class2', ...]
        :param tensorflow_version:
        """
        self.model = None
        self.graph = None
        self.session = None
        self.image_tensor = None
        self.detection_boxes = None
        self.detection_scores = None
        self.detection_classes = None
        self.num_detections = None
        if len(cls_labels) == 1:
            cls_labels = ["background"] + cls_labels
        self.cls_labels = cls_labels
        self.project_labels = cls_labels[1:]
        self.tensorflow_version = int(tf.__version__[0])
        self.gpu = gpu
        print(f"GPU: {gpu}")
        with tf.device(f"/gpu:{gpu}"):
            self.load_model(model_path)

    def load_model(self, model_path):
        """
        Loads an object detection model (this function is performed during init)

        TF1 - creates a session, graph and the relevant input and output tensors
        TF2 - creates a Model

        :param model_path: path to the saved_model directory (TF2) or frozen graph (.pb file TF1)
        """
        # Tensorflow Version 1
        if self.tensorflow_version == 1:
            graph = tf.Graph()
            with graph.as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile(model_path, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')
                sess = tf.Session(graph=graph)
            self.graph = graph
            self.session = sess
            self.image_tensor = graph.get_tensor_by_name('image_tensor:0')
            self.detection_boxes = graph.get_tensor_by_name('detection_boxes:0')
            self.detection_scores = graph.get_tensor_by_name('detection_scores:0')
            self.detection_classes = graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = graph.get_tensor_by_name('num_detections:0')

        # Tensorflow version 2
        elif self.tensorflow_version == 2:
            self.model = tf.saved_model.load(model_path)

    def detect(self, im, threshold):
        """
        Perform object detection on an image

        :param im: image to process
        :param threshold: detection threshold in range [0, 1]
        :return: list of detection results
        """
        with tf.device(f"/gpu:{self.gpu}"):
            h = im.shape[0]
            w = im.shape[1]
            im = im[np.newaxis, ...]
            results = []

            # Tensorflow Version 1
            if self.tensorflow_version == 1:
                (boxes, scores, classes, num) = self.session.run([self.detection_boxes,
                                                                  self.detection_scores,
                                                                  self.detection_classes,
                                                                  self.num_detections],
                                                                 feed_dict={self.image_tensor: im})
                for i, box in enumerate(boxes[0]):
                    score = scores[0][i]
                    if score > threshold:
                        ymin, xmin, ymax, xmax = box
                        det = dict()
                        det['x'] = int(xmin * w)
                        det['y'] = int(ymin * h)
                        det['width'] = int((xmax - xmin) * w)
                        det['height'] = int((ymax - ymin) * h)
                        det['label'] = self.cls_labels[int(classes[0][i])]
                        det['score'] = score.astype(np.float64)
                        det['im_width'] = w
                        det['im_height'] = h
                        results.append(det)

            # Tensorflow Version 2
            elif self.tensorflow_version == 2:
                detections = self.model(im)
                num = int(detections.pop('num_detections'))
                detections = {key: value[0, :num].numpy()
                              for key, value in detections.items()}
                classes = detections['detection_classes'].astype(np.int64)
                boxes = detections['detection_boxes']
                scores = detections['detection_scores']
                for i, box in enumerate(boxes):
                    score = scores[i]
                    if score > threshold:
                        ymin, xmin, ymax, xmax = box
                        det = dict()
                        det['x'] = int(xmin * w)
                        det['y'] = int(ymin * h)
                        det['width'] = int((xmax - xmin) * w)
                        det['height'] = int((ymax - ymin) * h)
                        det['label'] = self.cls_labels[int(classes[i])]
                        det['score'] = score.astype(np.float64)
                        det['im_width'] = w
                        det['im_height'] = h
                        det['track_id'] = None
                        det['track_idx'] = 0
                        det['track_len'] = 1
                        results.append(det)
            return results

    def detect_files(self,
                     filenames: OrderedDict,
                     threshold,
                     skip=1):
        """
        Perform object detection on a list of files

        :param filenames: list of image filenames (full paths)
        :param threshold: detection threshold in range [0, 1]
        :param skip: only process every `skip` images
        :return: Project containing the results
        """
        results = OrderedDict()
        pbar = tqdm(total=len(filenames))
        for idx, (key, filename) in enumerate(filenames.items()):
            if idx % skip != 0:
                continue
            try:
                im = skio.imread(filename)
            except:
                print(f"Error loading: {filename}")
                continue
            detections = self.detect(im, threshold)
            results[key] = dict()
            results[key]['filename'] = filename
            try:
                results[key]['size'] = os.path.getsize(filename)
            except:
                results[key]['size'] = 0
            results[key]['detections'] = detections
            if len(detections) > 0:
                pbar.set_description("DET {}".format(len(detections)))
            else:
                pbar.set_description("     ")
            pbar.update(skip)
        pbar.close()

        return ObjectDetector.detections_to_project(results)

    def detect_files_with_tracker(self,
                                  filenames: OrderedDict,
                                  threshold,
                                  iou_threshold=0.3,
                                  memory=15,
                                  subsample=4,
                                  trim=True):
        print("Detect files with tracker")
        print(f"threhsold {threshold}")
        print(f"iou_threshold {iou_threshold}")
        print(f"memory {memory}")
        print(f"subsample {subsample}")
        tracker = Tracker(flow_resample=subsample,
                          iou_threshold=iou_threshold,
                          memory=memory)

        pbar = tqdm(total=len(filenames))
        ds = ObjectDetector.filenames_dataset(filenames)
        for idx, (key, im) in enumerate(ds):
            im = im.numpy()
            key = key.numpy()
            # Skip image if only processing every n-th image
            start = time.time()
            # if idx % skip != 0:
            #     continue
            # # Load the image
            # try:
            #     im = skio.imread(filename)
            # except:
            #     print(f"Error loading: {filename}")
            #     continue
            # load_time = time.time()
            # Perform detection
            detections = self.detect(im, threshold)
            det_time = time.time()
            # Convert to annotations
            annotations = ObjectDetector.detections_to_track_annotations(idx, detections)
            # Update tracker
            tracker.update(im, annotations)
            track_time = time.time()
            pbar.update(1)
            pbar.set_description(
                f"DET {len(detections):02d} TRK {len(tracker.active_tracks):02d} det {det_time - start:.03f}s trk {track_time - det_time:.03f}s")
        pbar.close()

        # Trim ends of tracks
        if trim:
            tracker.trim(keypoint_threshold=3, max_propagated=5)
        return ObjectDetector.tracker_to_project(filenames, tracker)

    def detect_directory(self,
                         input_dir,
                         threshold,
                         skip=1,
                         extension="*.jpg",
                         use_tracker=True,
                         tracker_iou_threshold=0.3,
                         tracker_memory=5,
                         tracker_subsample=4,
                         tracker_trim=True,
                         offset=0):
        """
        Perform object detection on a directory of images

        :param input_dir: input directory containing images
        :param threshold: detection threshold in range [0, 1]
        :param skip: only process every `skip` images
        :param extension: filename extension (default is *.jpg)
        :return: Project with detections
        """
        filenames = OrderedDict()
        for idx, fn in enumerate(sorted(glob(os.path.join(input_dir, extension)))):
            if isinstance(offset, list) or isinstance(offset, tuple):
                if idx < offset[0] or idx > offset[1]:
                    continue
            else:
                if idx < offset:
                    continue
            if idx % skip == 0:
                filenames[str(Path(fn).stem)] = fn
        if use_tracker:
            return self.detect_files_with_tracker(filenames,
                                                  threshold,
                                                  tracker_iou_threshold,
                                                  tracker_memory,
                                                  tracker_subsample,
                                                  tracker_trim)
        else:
            return self.detect_files(filenames, threshold, skip)

    def detect_project(self,
                       project: Project,
                       threshold,
                       skip=1,
                       use_tracker=False,
                       tracker_iou_threshold=0.3,
                       tracker_memory=15,
                       tracker_subsample=4,
                       tracker_trim=True,
                       offset=0):
        """
        Perform object detection on a Project object
        :param project: Project object
        :param threshold: detection threshold in range [0, 1]
        :param skip: only process every `skip` images
        :return: Project with detections
        """
        # Detect using a very small threshold (0.01), needed for the coco metrics
        filenames = OrderedDict()
        for idx, (key, metadata) in enumerate(project.image_dict.items()):
            if isinstance(offset, list) or isinstance(offset, tuple):
                if idx < offset[0] or idx > offset[1]:
                    continue
            else:
                if idx < offset:
                    continue
            filenames[key] = metadata.filename
        if use_tracker:
            result = self.detect_files_with_tracker(filenames,
                                                    threshold,
                                                    tracker_iou_threshold,
                                                    tracker_memory,
                                                    tracker_subsample,
                                                    tracker_trim)
        else:
            result = self.detect_files(filenames, threshold, skip)

        # COCO evaluation
        coco_evaluate(project, result, self.project_labels)

        # Apply treshold
        result.remove_annotations_below_threshold(threshold)

        # F2 evaluations
        all_f2s = []
        for iou in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.80]:
            f2 = f_beta_score(project, result, iou, 0.01, beta=2)
            all_f2s.append(f2[0])
            print(f"IoU: {iou:.02f}, F2 score: {f2[0]:.03f}, TP: {f2[1]}, FP: {f2[2]}, FN: {f2[3]}")
        print(f"Mean F2 score: {np.mean(all_f2s):.03f}")
        return result

    def detect_project_file(self,
                            project_file,
                            threshold,
                            skip=1,
                            use_tracker=False,
                            tracker_iou_threshold=0.3,
                            tracker_memory=15,
                            tracker_subsample=4,
                            tracker_trim=True,
                            offset=0):
        project = Project.load(project_file)
        return self.detect_project(project,
                                   threshold,
                                   skip=skip,
                                   use_tracker=use_tracker,
                                   tracker_iou_threshold=tracker_iou_threshold,
                                   tracker_memory=tracker_memory,
                                   tracker_subsample=tracker_subsample,
                                   tracker_trim=tracker_trim,
                                   offset=offset)

    @staticmethod
    def detections_to_project(detections):
        """
        Convert detections to a Project
        :param detections: list of detections
        :return: new Project with detections
        """
        detected_project = Project()
        for key, detection in detections.items():
            metadata = ImageMetadata(detection['filename'], detection['size'])
            annotations = []
            for d in detection['detections']:
                annotation = RectangleAnnotation(d['x'],
                                                 d['y'],
                                                 d['width'],
                                                 d['height'],
                                                 d['label'],
                                                 d['score'],
                                                 "machine",
                                                 seq_id=d['track_id'],
                                                 seq_idx=d['track_idx'],
                                                 seq_len=d['track_len'])
                annotations.append(annotation)
            metadata.annotations = annotations
            detected_project.add_image(metadata, key=key)
        return detected_project

    @staticmethod
    def detections_to_annotations(detections):
        annotations = []
        for d in detections:
            annotation = RectangleAnnotation(d['x'],
                                             d['y'],
                                             d['width'],
                                             d['height'],
                                             d['label'],
                                             d['score'],
                                             "machine",
                                             seq_id=d['track_id'],
                                             seq_idx=d['track_idx'],
                                             seq_len=d['track_len'])
            annotations.append(annotation)
        return annotations

    @staticmethod
    def detections_to_track_annotations(frame_id, detections):
        annotations = []
        for d in detections:
            annotation = TrackAnnotation(frame_id,
                                         d['x'],
                                         d['y'],
                                         d['width'],
                                         d['height'],
                                         d['label'],
                                         d['score'],
                                         True)
            annotations.append(annotation)
        return annotations

    @staticmethod
    def tracker_to_project(filenames: OrderedDict, tracker: Tracker):
        project = Project()
        keys = list(filenames.keys())
        for key, filename in filenames.items():
            metadata = ImageMetadata(filename)
            project.add_image(metadata, key)

        for seq_id, track in enumerate(tracker.tracks):
            for seq_idx, ann in enumerate(track.annotations):
                key = keys[ann.frame_id]
                metadata = project.image_dict[key]
                new_ann = RectangleAnnotation(ann.x,
                                              ann.y,
                                              ann.width,
                                              ann.height,
                                              ann.label,
                                              ann.score,
                                              seq_id=seq_id,
                                              seq_idx=seq_idx,
                                              seq_len=len(track.annotations),
                                              frame_id=ann.frame_id,
                                              is_keypoint=ann.is_keypoint)
                metadata.add_annotation(new_ann)

        return project

    @staticmethod
    def filenames_dataset(filenames: OrderedDict):
        keys = list(filenames.keys())
        fns = list(filenames.values())
        ds = tf.data.Dataset.from_tensor_slices((keys, fns))
        ds = ds.map(ObjectDetector.map_fn, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.prefetch(32)
        return ds

    @staticmethod
    def map_fn(key, fn):
        im = tf.io.read_file(fn)
        im = tf.io.decode_image(im)
        return key, im


if __name__ == "__main__":
    # det = ObjectDetector(
    #     "/media/mar76c/DATA/Tracker_test/ssd_mobilenet_v2_1280/saved_model",
    #     ["background", "COTS_NV"])
    # project = det.detect_directory("/media/mar76c/DATA/Tracker_test/test_pos", 0.2,
    #                                use_tracker=True,
    #                                tracker_memory=15,
    #                                tracker_subsample=2,
    #                                tracker_trim=True,
    #                                offset=0)
    # project.save_as("odtest.json", indent=4)
    # project.save_as("odtest.json", format="via", indent=4)

    det = ObjectDetector(
        "/media/mar76c/DATA/Tracker_test/ssd_mobilenet_v2_1280/saved_model",
        ["background", "COTS"])
    project = Project.load("/mnt/ssd/code/visage-ml-tf2objdet/annotations/Kaggle_720_v4/split_test.json")
    project.update_image_directory_start("/home/aslab", "/mnt/ssd")
    project = det.detect_project(
        project,
        0.1,
        use_tracker=True,
        tracker_iou_threshold=0.2,
        tracker_memory=15,
        tracker_subsample=2,
        tracker_trim=False
    )
    project.save_as("kaggle.json", indent=4)
    project.save_as("kaggle_via.json", format="via", indent=4)
