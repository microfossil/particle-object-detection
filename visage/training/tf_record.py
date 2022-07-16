"""
Generates tf records from a VIA project file

Splits according to sequence (i.e, sequence of images will remain in either test/train and will not
be mixed in either)
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import shutil

import tensorflow as tf
from skimage.color import rgb2gray
from tqdm import tqdm
import numpy as np
import skimage.io as skio
import skimage.transform as skt
import imageio

from visage.project.image import ImageMetadata
from visage.project.project import Project


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def read_examples_list(path):
    with tf.gfile.GFile(path) as fid:
        lines = fid.readlines()
    return [line.strip().split(' ')[0] for line in lines]


def recursive_parse_xml_to_dict(xml):
    if not xml:
        return {xml.tag: xml.text}
    result = {}
    for child in xml:
        child_result = recursive_parse_xml_to_dict(child)
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


def create_tf_example(metadata: ImageMetadata,
                      cls_labels: list = None,
                      skip_if_no_annotations=True,
                      resize_height=None,
                      as_greyscale=False,
                      save_file_as=None):
    """
    Creates a single record example from image and its annotations
    :param metadata: the annotated image
    :param cls_labels: valid class labels to include (None for all)
    :param skip_if_no_annotations: skip this record if there are no valid annotations
    :param resize_height: resize image to have this height
    :param save_file_as: save the file as well
    :return:
    """
    # Skip this example if no annotations
    if skip_if_no_annotations:
        found_annotation = False
        if cls_labels is None and len(metadata.annotations) > 0:
            found_annotation = True
        else:
            for ann in metadata.annotations:
                if ann.label in cls_labels:
                    found_annotation = True
        if found_annotation is False:
            return None

    # Load the image
    with tf.io.gfile.GFile(metadata.filename, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)

    try:
        image = imageio.imread(encoded_jpg_io, format='jpg')
        if as_greyscale:
            image = rgb2gray(image)
            image = np.round(image * 255).astype(np.uint8)
            image = np.repeat(image[..., np.newaxis], 3, axis=-1)
    except:
        print(f"Image: {metadata.filename} could not be read")
        return None

    # Resize if necessary
    was_resized = False
    original_height = image.shape[0]
    original_width = image.shape[1]
    if resize_height is not None and resize_height > 0:
        factor = resize_height / original_height
        height = int(np.round(original_height * factor))
        width = int(np.round(original_width * factor))
        if height != original_height:
            image = np.round(skt.resize(image, (height, width, 3)) * 255).astype(np.uint8)
            was_resized = True
            # image = image.resize((width, height))
            if save_file_as is not None:
                skio.imsave(save_file_as, image, check_contrast=False, quality=95)
                # image.save(save_file_as, quality=95)
        else:
            if save_file_as is not None:
                shutil.copy(metadata.filename, save_file_as)
    else:
        width = original_width
        height = original_height
        if save_file_as is not None:
            shutil.copy(metadata.filename, save_file_as)

    # Encode image if was changed
    if was_resized or as_greyscale:
        temp = io.BytesIO()
        imageio.imwrite(temp, image, format='jpg', quality=95)
        encoded_jpg = temp.getvalue()

    # Encode the information
    filename = metadata.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    # Add annotations if in the class list
    for annotation in metadata.annotations:
        if annotation.label in cls_labels:
            cls_idx = cls_labels.index(annotation.label) + 1
            cls_label = annotation.label
        else:
            continue
        classes_text.append(cls_label.encode())
        classes.append(cls_idx)
        xmins.append(annotation.x / original_width)
        ymins.append(annotation.y / original_height)
        xmaxs.append((annotation.x + annotation.width) / original_width)
        ymaxs.append((annotation.y + annotation.height) / original_height)

    # if len(xmins) > 0:
    #     if np.min(xmins) < 0 or np.max(xmaxs) > 1 or np.min(ymins) < 0 or np.max(ymaxs) > 1:
    #         print("Annotation out of range")
    #         print(f"{original_width} {original_height} - {width} {height}")
    #         print(filename)
    #         print(xmins)
    #         print(xmaxs)
    #         print(ymins)
    #         print(ymaxs)
    #         for ann in metadata.annotations:
    #             print(f"{ann.x} {ann.x+ann.width} {ann.y} {ann.y+ann.height}")

    # print(encoded_jpg)

    # Create the tf example
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/filename': bytes_feature(filename),
        'image/source_id': bytes_feature(filename),
        'image/encoded': bytes_feature(encoded_jpg),
        'image/format': bytes_feature(image_format),
        'image/object/bbox/xmin': float_list_feature(xmins),
        'image/object/bbox/xmax': float_list_feature(xmaxs),
        'image/object/bbox/ymin': float_list_feature(ymins),
        'image/object/bbox/ymax': float_list_feature(ymaxs),
        'image/object/class/text': bytes_list_feature(classes_text),
        'image/object/class/label': int64_list_feature(classes),
    }))
    return tf_example


def create_tf_record(project: Project,
                     output_file,
                     cls_labels=None,
                     skip_if_no_annotations=True,
                     save_images_in=None,
                     resize_height=None,
                     as_greyscale=True
                     ):
    """
    Creates a TF record from a project
    :param project: project object
    :param output_file: file to save TF record
    :param cls_labels: the class labels that will be include (None for all from project)
    :param skip_if_no_annotations: skip adding the record if there are no annotations
    :return:
    """
    if os.path.dirname(output_file) != "":
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if cls_labels is None:
        cls_labels = list(project.label_dict.keys())

    # image_metadata = [v for k, v in project.image_dict.items()]
    # proj_labels = [k for k, v in project.label_dict.items()]

    # print('-' * 80)
    # print("Create TF record")
    # print('-' * 80)
    # print("Project: {}".format(project.filename))
    # print("Output file: {}".format(output_file))
    # print(" - {} images".format(len(image_metadata)))
    # print(" - class labels {}".format(proj_labels))
    # print(" - valid class labels {}".format(cls_labels))

    skipped_count = 0
    annotated_count = 0

    data_writer = tf.io.TFRecordWriter(output_file)
    # print()
    # print("Generate records...")
    for key, metadata in tqdm(project.image_dict.items()):
        if save_images_in:
            save_filename = os.path.join(save_images_in, key + ".jpg")
        else:
            save_filename = None
        record = create_tf_example(metadata,
                                   cls_labels,
                                   skip_if_no_annotations=skip_if_no_annotations,
                                   resize_height=resize_height,
                                   as_greyscale=as_greyscale,
                                   save_file_as=save_filename)
        if record is not None:
            data_writer.write(record.SerializeToString())
            annotated_count += 1
        else:
            skipped_count += 1
    data_writer.close()
