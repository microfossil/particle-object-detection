import os
import shutil
from collections import OrderedDict
from typing import List

import pandas
import pandas as pd
from skimage.color import rgb2gray
from tqdm import tqdm
import skimage.transform as skt
import skimage.io as skio
import numpy as np

from visage.cvat.project_api import CVAT
from visage.project.project import Project
from visage.training.tf_record import create_tf_record
from visage.training.train_val_split import split_project_by_sequence

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

def create_training_dataset(train_val_projects: List[Project],
                            test_projects: List[Project],
                            output_dir,
                            tp_labels=None,
                            tn_labels=None,
                            validation_split=0.1,
                            tn_tp_ratio=1,
                            resize_height=None,
                            create_tf_records=True,
                            save_images=False,
                            as_greyscale=False,
                            random_seed=0):
    """
    Creates the train, validation and test datasets from a set of projects
    - train contains only with true positives (TPs) and either those with true negative (TN) labels or those sampled from the remaining images
    - val contains only with TP and either those with TN labels or those sampled from the remaining images
    - test contains ALL images in the test project

    Args:
        train_val_projects: the projects that will be combined then split into training and validation
        test_projects: the projects that will be combined into test
        output_dir: directory to store the json files and tf records
        tp_labels: annotations with these TP labels and the corresponding image will be included, images without any of these annotations are excluded by default
        tn_labels: images with only these TN annotations will have the annotations removed and be included as (hard) negatives
        validation_split: percentage split (of sequences if selected, otherwise images) in the validation set
        tn_tp_ratio: number of TN images to include as a multiple of the number of TP images, if 0 all are included
        resize_height: resize the images to have this height
        create_tf_records: create TF record fies for the train, validation and test splits
        save_images: save the (resized) images to the annotations directory as well
        as_greyscale: convert images to greyscale
        random_seed: set to a fixed value to get the same split each time (0 by default)
    """
    # Make output directory if not already
    os.makedirs(output_dir, exist_ok=True)

    # Combine train / val projects
    train_val = Project()
    for i, project in enumerate(train_val_projects):
        train_val.append_project(project, prefix=f"{i:04d}")
    original_train_val = train_val.copy()
    original_train_val.keep_annotations_with_labels(tp_labels)

    # Create train and validation sets
    # - all TP images are used
    # - a random selection of TN images are used
    train, val = split_project_by_sequence(train_val,
                                           tp_labels=tp_labels,
                                           tn_labels=tn_labels,
                                           val_split=validation_split,
                                           random_seed=random_seed,
                                           tn_tp_ratio=tn_tp_ratio)
    train.keep_annotations_with_labels(tp_labels)
    val.keep_annotations_with_labels(tp_labels)

    if tp_labels is None:
        tp_labels = train_val.label_dict.keys()

    # Combine test projects
    test = Project()
    for i, project in enumerate(test_projects):
        test.append_project(project, prefix=f"{i:04d}")
    test.keep_annotations_with_labels(tp_labels)
    original_test = test.copy()

    # Save TF records if desired
    if create_tf_records:
        print()
        print("Saving TF records...")
        print("- train")
        create_tf_record(train,
                         os.path.join(output_dir, "train.tfrecord"),
                         cls_labels=None,
                         skip_if_no_annotations=False,
                         resize_height=resize_height,
                         as_greyscale=as_greyscale)
        print("- val")
        create_tf_record(val,
                         os.path.join(output_dir, "val.tfrecord"),
                         cls_labels=None,
                         skip_if_no_annotations=False,
                         resize_height=resize_height,
                         as_greyscale=as_greyscale)
        print("- test")
        create_tf_record(test,
                         os.path.join(output_dir, "test.tfrecord"),
                         cls_labels=None,
                         skip_if_no_annotations=False,
                         resize_height=resize_height,
                         as_greyscale=as_greyscale)

    # Save images if desired
    if save_images:
        os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "val"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)

        # Export the images (also adjusts size)
        print()
        print("Exporting images...")
        print(" - train")
        export_images(train, output_dir, "train", resize_height, as_greyscale)
        print(" - val")
        export_images(val, output_dir, "val", resize_height, as_greyscale)
        print(" - test")
        export_images(test, output_dir, "test", resize_height, as_greyscale)

        # Update the paths to point to the saved images
        train.update_image_directory(os.path.join(output_dir, "train"))
        val.update_image_directory(os.path.join(output_dir, "val"))
        test.update_image_directory(os.path.join(output_dir, "test"))

        # Change image key to the filename to prevent conflict as GoPro runs have the same filenames
        train.update_image_filename_from_key()
        val.update_image_filename_from_key()
        test.update_image_filename_from_key()

    # Save label map
    with open(os.path.join(output_dir, "label_map.txt"), "w") as label_map:
        for idx, label in enumerate(tp_labels):
            label_map.write("item {{\n    id: {}\n    name: \"{}\"\n}}\n".format(idx + 1, label))

    # Save original projects
    print()
    print("Saving project files...")
    print("- original train + val")
    original_train_val.metadata["name"] = "original_train_val"
    original_train_val.save_as(os.path.join(output_dir, "source_all_train+val.json"))
    original_train_val.save_as(os.path.join(output_dir, "via_source_all_train+val.json"), format='via')
    original_train_val.save_as(os.path.join(output_dir, "source_all_train+val.csv"), format='mlboxv2')
    print("- original test")
    original_test.metadata["name"] = "original_test"
    original_test.save_as(os.path.join(output_dir, "source_all_test.json"))
    original_test.save_as(os.path.join(output_dir, "via_source_all_test.json"), format='via')
    original_test.save_as(os.path.join(output_dir, "source_all_test.csv"), format='mlboxv2')

    # Save project splits
    print("- train")
    train.metadata["name"] = "train"
    train.save_as(os.path.join(output_dir, "split_train.json"))
    train.save_as(os.path.join(output_dir, "via_split_train.json"), format='via')
    train.save_as(os.path.join(output_dir, "split_train.csv"), format='mlboxv2')
    print("- val")
    val.metadata["name"] = "val"
    val.save_as(os.path.join(output_dir, "split_val.json"))
    val.save_as(os.path.join(output_dir, "via_split_val.json"), format='via')
    val.save_as(os.path.join(output_dir, "split_val.csv"), format='mlboxv2')
    print("- test")
    test.metadata["name"] = "test"
    test.save_as(os.path.join(output_dir, "split_test.json"))
    test.save_as(os.path.join(output_dir, "via_split_test.json"), format='via')
    test.save_as(os.path.join(output_dir, "split_test.csv"), format='mlboxv2')

    return train, val, test


def create_training_set_from_cvat_tasks(cvat_url,
                                        train_val: list,
                                        test: list,
                                        output_dir,
                                        tp_labels=['COTS'],
                                        tn_labels=['COTS_FP', 'No_COTS'],
                                        split=0.1,
                                        tn_tp_ratio=1,
                                        create_tf_records=True,
                                        save_images=False,
                                        resize_height=None,
                                        as_greyscale=False,
                                        local_dir="/mnt/ssd",
                                        cvat_dir=""):
    # Hack for FN shenanigans where some of the true labels have been labelled as false negatives
    # Make sure every true positive label also has the label_FN form in there as well
    new_tp_labels = []
    fn_labels = []
    for label in tp_labels:
        if not label.endswith("_FN"):
            new_tp_labels.append(label)
            fn_labels.append(label + "_FN")
    tp_labels = new_tp_labels

    # Ensure list not tuple
    tn_labels = list(tn_labels)

    # Load CVAT
    cvat_projects = CVAT(cvat_url)
    cvat_projects.load()

    # Load training and validation datasets
    train_val_projects = []
    for task_proj in train_val:
        task = cvat_projects.load_task_by_code(task_proj)
        project = task.project
        project.keep_annotations_with_labels(tp_labels + fn_labels + tn_labels)
        # Change any annotations labels with FN in there back to normal annotations
        for label in fn_labels:
            project.update_label(label, label[:-3])
        if local_dir is not None:
            project.update_image_directory_start(cvat_dir, local_dir)
        train_val_projects.append(project)

    # Load test projects
    test_projects = []
    for task_proj in test:
        task = cvat_projects.load_task_by_code(task_proj)
        project = task.project
        for key, val in project.image_dict.items():
            print(val.filename)
            break
        project.keep_annotations_with_labels(tp_labels + fn_labels + tn_labels)
        # Change FN to normal label...
        for label in fn_labels:
            project.update_label(label, label[:-3])
        if local_dir is not None:
            project.update_image_directory_start(cvat_dir, local_dir)
        for key, val in project.image_dict.items():
            print(val.filename)
            break
        test_projects.append(project)

    # Save summary
    save_projects_summary([train_val_projects, test_projects],
                          ["source_train_val", "source_test"],
                          output_dir,
                          tp_labels,
                          tn_labels)

    # Create the training dataset from the projects
    print(train_val_projects)
    print(test_projects)
    train, val, test = create_training_dataset(train_val_projects,
                                               test_projects,
                                               output_dir,
                                               tp_labels,
                                               tn_labels,
                                               validation_split=split,
                                               tn_tp_ratio=tn_tp_ratio,
                                               resize_height=resize_height,
                                               create_tf_records=create_tf_records,
                                               save_images=save_images,
                                               as_greyscale=as_greyscale,
                                               random_seed=0)

    # Save summary
    save_projects_summary([[train], [val], [test]],
                          ["train", "val", "test"],
                          output_dir,
                          tp_labels,
                          tn_labels)

    # Create some VIA project files to enable remote viewing
    if local_dir.startswith("/mnt/nas") or local_dir.startswith("/mnt/ssd"):
        start = local_dir[:8]
        if local_dir.startswith("/mnt/ssd"):
            port = 8000
        else:
            port = 8001
        train.update_image_directory_start(start, f"http://cruncher-ph.nexus.csiro.au:{port}")
        train.save_as(os.path.join(output_dir, "via_split_train_remote.json"), "via")
        val.update_image_directory_start(start, f"http://cruncher-ph.nexus.csiro.au:{port}")
        val.save_as(os.path.join(output_dir, "via_split_val_remote.json"), "via")
        test.update_image_directory_start(start, f"http://cruncher-ph.nexus.csiro.au:{port}")
        test.save_as(os.path.join(output_dir, "via_split_test_remote.json"), "via")


def save_projects_summary(project_lists, names, output_dir, tp_labels, tn_labels):
    for i, projects in enumerate(project_lists):
        print()
        print("=" * 80)
        print(names[i])
        labels = tp_labels + tn_labels
        labels_df, sequences_df = projects_summary_as_dataframe(projects, labels)
        print("-" * 80)
        print("Label count")
        print(labels_df)
        print(labels_df.sum())
        print("-" * 80)
        print("Track count")
        print(sequences_df)
        print(sequences_df.sum())
        # Store summary in file
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"{names[i]}.txt"), "w") as f:
            f.write("Labels\n")
            print(labels_df, file=f)
            f.write("\nLabels summary\n")
            print(labels_df.iloc[:, 1:].sum(), file=f)
            f.write("\nTracks\n")
            print(sequences_df, file=f)
            f.write("\nTracks summary\n")
            print(sequences_df.iloc[:, 1:].sum(), file=f)


def projects_summary_as_dataframe(projects, labels):
    total_labels = []
    total_tracks = []
    for project in projects:
        label_count = project.label_count()
        image_with_label_count = project.image_with_label_count()
        this_labels = OrderedDict()
        this_labels["Survey"] = project.metadata["name"]
        this_labels["images"] = len(project.image_dict)
        this_labels.update({label: label_count.get(label) or 0 for label in labels})
        this_labels.update({"images_" + label: image_with_label_count.get(label) or 0 for label in labels})
        total_labels.append(this_labels)

        sequence_count = project.sequence_count()
        this_sequences = OrderedDict()
        this_sequences["Survey"] = project.metadata["name"]
        this_sequences.update({label: sequence_count.get(label) or 0 for label in labels})
        total_tracks.append(this_sequences)

    labels_df = pandas.DataFrame(total_labels)
    sequences_df = pandas.DataFrame(total_tracks)

    return labels_df, sequences_df


def export_images(project, output_dir, split_name, resize_height=None, as_greyscale=False):
    print(as_greyscale)
    save_directory = os.path.join(output_dir, split_name)
    os.makedirs(save_directory, exist_ok=True)
    for key, metadata in tqdm(project.image_dict.items()):
        # Resize the image if necessary
        if (resize_height is not None and resize_height > 0) or as_greyscale:
            im = skio.imread(metadata.filename)
            # 3 channels greyscale
            if as_greyscale:
                im = rgb2gray(im)
                im = np.round(im * 255).astype(np.uint8)
                im = np.repeat(im[..., np.newaxis], 3, axis=-1)
            if (resize_height is not None and resize_height > 0):
                factor = (resize_height / im.shape[0])
                new_width = int(im.shape[1] * factor)
                if new_width != im.shape[1]:
                    im = np.round(skt.resize(im, [resize_height, new_width, im.shape[2]]) * 255)
                    im = im.astype(np.uint8)
                    metadata.rescale_annotation(factor)
            skio.imsave(os.path.join(output_dir, split_name, key + ".jpg"), im, check_contrast=False, quality=95)
        # Otherwise just copy it
        else:
            shutil.copy(metadata.filename, os.path.join(output_dir, split_name, key + ".jpg"))


def create_google_dataframe(project: Project, height, width, ml_use) -> pd.DataFrame:
    rows = []
    for key, metadata in project.image_dict.items():
        if len(metadata.annotations) > 0:
            for ann in metadata.annotations:
                vals = []
                vals.append(ml_use)
                vals.append(metadata.filename)
                vals.append(ann.label)
                vals.append(ann.x / width)
                vals.append(ann.y / height)
                vals.append("")
                vals.append("")
                vals.append((ann.x + ann.width) / width)
                vals.append((ann.y + ann.height) / height)
                vals.append("")
                vals.append("")
                rows.append(vals)
        else:
            vals = []
            vals.append(ml_use)
            vals.append(metadata.filename)
            vals.append("")
            vals.append("")
            vals.append("")
            vals.append("")
            vals.append("")
            vals.append("")
            vals.append("")
            vals.append("")
            vals.append("")
            rows.append(vals)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Examples

    # Train / val sets
    train_val_202007 = ["20200718_heron_dive_morings_v4@Heron Island July 2020",
                        "20200716_fitzroy_ff_v4@Heron Island July 2020",
                        "20200716_fitzroy_channel_west_v4@Heron Island July 2020",
                        "20200712_fitzroy_east_to_front_v4@Heron Island July 2020",
                        "20200712_fitzroy_lagoon_to_east_2_v4@Heron Island July 2020",
                        "20200717_fitzroy_shaol_v4@Heron Island July 2020",
                        "20200716_fitzroy_east_to_north_repeat_v4@Heron Island July 2020",
                        "20200717_llewellyn2_v4@Heron Island July 2020"]

    train_val_202106 = ["20210618_heron_island_channel_q@Heron Island June 2021",
                        "20210621_heron_north_east_q@Heron Island June 2021",
                        "20210623_heron_north_east_2_q@Heron Island June 2021",
                        "20210624_fitzroy@Heron Island June 2021",
                        "20210624_fitzroy_2@Heron Island June 2021"]

    # Test sets
    test_202007 = ["20200716_llewellyn_south_west_corner_v4@Heron Island July 2020",
                   "20200717_lamont_1_v4@Heron Island July 2020"]

    test_202106 = ["20210619_lamont_q@Heron Island June 2021",
                   "20210620_one_tree_q@Heron Island June 2021"]

    # Path to save
    annotations_path = "/mnt/ssd/training_sets"

    create_training_set_from_cvat_tasks("http://cruncher-ph.nexus.csiro.au:8080",
                                        train_val=train_val_202007 + train_val_202106,
                                        test=test_202106 + test_202007,
                                        output_dir=f"{annotations_path}/heron_combined",
                                        create_tf_records=False)
