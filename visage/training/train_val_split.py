import re
from scipy.ndimage import binary_dilation
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from visage.project.image import ImageMetadata
from visage.project.project import Project


def split_project_by_sequence(project: Project,
                              tp_labels,
                              tn_labels,
                              val_split=0.2,
                              random_seed=0,
                              tn_tp_ratio=1,
                              buffer=15,
                              sequence_gap=5):
    print("-" * 80)
    print("Split project")
    print(f"- project labels: {project.label_count()}")
    print(f"- TP labels: {tp_labels}")
    print(f"- TN labels: {tn_labels}")
    print(f"- split: {val_split}")
    print(f"- TN to TP ratio: {tn_tp_ratio}")
    print(f"- # frame buffer after sequence: {buffer}")

    # Create index for frame type
    # 0 - no annotation
    # 1 - contains TP
    # 2 - buffer
    # 3 - contains FP (but no TP)

    # Allocate array
    frame_type = np.zeros(len(project.image_dict), dtype=np.int32)

    # Find TP and TN
    for i, (key, metadata) in enumerate(project.image_dict.items()):
        if metadata.has_labels(tp_labels):
            frame_type[i] = 1
        elif metadata.has_labels(tn_labels):
            frame_type[i] = 3

    # print(f"total tp frames: {np.sum(frame_type == 1)}")
    # print(f"total tn frames: {np.sum(frame_type == 3)}")

    # Find buffer
    buffer_frames = binary_dilation((frame_type == 1), structure=np.ones(buffer * 2 + 1))
    buffer_frames = np.logical_xor(buffer_frames, frame_type == 1)
    frame_type[buffer_frames] = 2

    # print(f"total tp frames: {np.sum(frame_type == 1)}")
    # print(f"total buffer frames: {np.sum(frame_type == 2)}")
    # print(f"total tn frames: {np.sum(frame_type == 3)}")

    # Randomly select TN frames
    # Number of TPs
    TP_idxs = np.where(frame_type == 1)[0]

    # Max number of TNs
    TN_to_choose = int(len(TP_idxs) * tn_tp_ratio)
    TN_idxs = []

    # Select from FP frames first
    FP_idxs = np.where(frame_type == 3)[0]
    if TN_to_choose >= len(FP_idxs):
        TN_idxs.extend(FP_idxs)
        TN_to_choose -= len(FP_idxs)

        # If still some left to choose, select from unlabelled
        unlab_idxs = np.where(frame_type == 0)[0]
        TN_to_choose = np.min((len(unlab_idxs), TN_to_choose))
        TN_idxs.extend(np.random.choice(unlab_idxs, TN_to_choose, replace=False))
    else:
        TN_idxs.extend(np.random.choice(FP_idxs, TN_to_choose, replace=False))

    # Sanity check
    print("Sanity check")
    print(f"- # TP {len(TP_idxs)}")
    print(f"- # TN {len(TN_idxs)}")

    # Create sequences
    # True positives
    prev_idx = -1
    TP_sequences = []
    current_TP_sequence = []
    for idx in TP_idxs:
        # A sequence has finished if the gap since the last index is large enough
        if idx > prev_idx + sequence_gap:
            TP_sequences.append(current_TP_sequence)
            current_TP_sequence = []
        current_TP_sequence.append(idx)
        prev_idx = idx
    if len(current_TP_sequence) > 0:
        TP_sequences.append(current_TP_sequence)

    # True negatives
    prev_idx = -1
    TN_sequences = []
    current_TN_sequence = []
    for idx in TN_idxs:
        # A sequence has finished if the gap since the last index is large enough
        if idx > prev_idx + sequence_gap:
            TN_sequences.append(current_TN_sequence)
            current_TN_sequence = []
        current_TN_sequence.append(idx)
        prev_idx = idx
    if len(current_TN_sequence) > 0:
        TN_sequences.append(current_TN_sequence)

    # Split sequences
    if val_split > 0:
        train_TP_sequences, val_TP_sequences = train_test_split(TP_sequences, test_size=val_split, random_state=random_seed)
        train_TN_sequences, val_TN_sequences = train_test_split(TN_sequences, test_size=val_split, random_state=random_seed)
    else:
        train_TP_sequences = TP_sequences
        val_TP_sequences = []
        train_TN_sequences = TN_sequences
        val_TN_sequences = []

    # Combined the sequences
    train_sequences = sorted([item for sublist in train_TP_sequences + train_TN_sequences for item in sublist])
    val_sequences = sorted([item for sublist in val_TP_sequences + val_TN_sequences for item in sublist])

    # Add to project
    metadata_dict = {i: (k, v) for i, (k, v) in enumerate(project.image_dict.items())}
    train_project = Project()
    val_project = Project()

    for idx in train_sequences:
        k, v = metadata_dict[idx]
        train_project.add_image(v, key=k)

    for idx in val_sequences:
        k, v = metadata_dict[idx]
        val_project.add_image(v, key=k)

    train_project.update_label_dict_from_metadata()
    val_project.update_label_dict_from_metadata()

    return train_project, val_project


def split_project_by_sequence_old(project: Project,
                                  tp_labels,
                                  tn_labels,
                                  val_split=0.2,
                                  random_seed=0,
                                  tn_only_every=1,
                                  negative_sampling=0,
                                  min_tp_sequence_gap=2,
                                  min_tn_sequence_gap=5):
    # TODO needs huge overhaul
    """
    Splits up the project into train and validation projects, removing all images that do not have the desired annotations

    :param project: object detection project
    :param tp_labels: images with these labels will be added to the split projects
    :param tn_labels: images with these labels will be added to the split projects, BUT WITH THESE ANNOTATIONS REMOVED
    :param val_split: fraction of project images to reserve for validation
    :param random_seed: random seed (set to fixed value to get same split)
    :param tn_only_every: if using true negatives as set by tn_labels, this will only add a TN image after this many images
    :param negative_sampling: if > 0  this will sample images without annotations as negative images.
            The number of images sampled will be the number of TP images times this value
            If -1, all negatives will be used
    :param min_tn_sequence_gap: minimum gap in frame until a true positive is considered not part of a sequence
    :param min_tp_sequence_gap: minimum gap in frame until a true negative is considered not part of a sequence
    :return: (train project, validation project)
    """
    print("-" * 80)
    print("Split project")
    print(f"- project labels: {project.label_count()}")
    print(f"- true positive labels: {tp_labels}")
    print(f"- true negative labels: {tn_labels}")

    tp_counts = 0
    tn_counts = 0

    # Nested list of tp sequences
    tp_sequences = []
    tp_current_sequence = []
    tp_sequence_count = 0
    tp_prev_frame = -min_tp_sequence_gap

    # Boolean which checks whether there are any tn labels
    print(negative_sampling)
    TN_PRESENT = len(tn_labels) != 0 and negative_sampling == 0
    TP_ALL = len(tp_labels) == 0

    # Nested list of tn sequences
    tn_sequences = []
    tn_current_sequence = []
    tn_sequence_count = 0
    tn_prev_frame = -min_tn_sequence_gap

    # tn images not explicitly labelled as such
    negative_images = []

    project.summary()

    # Create a project with just the images we have selection (tp_labels and tn_labels)
    for idx, metadata in tqdm(project.image_dict.items()):
        # Add the image and any annotations
        if TP_ALL and len(metadata.annotations) > 0:
            new_annotations = []
            for ann in metadata.annotations:
                new_annotations.append(ann)
            new_metadata = ImageMetadata(metadata.filename, metadata.filesize)
            new_metadata.annotations = new_annotations
            tp_counts += 1

            # frame = int(new_metadata.filename.split('_')[-1].replace(".jpg", ""))
            frame = int(re.search('(\d+)(?!.*\d)', new_metadata.filename).group(0))

            # Create new nested list if frame is part of new sequence
            if frame < tp_prev_frame:
                tn_prev_frame = -min_tp_sequence_gap
            if frame - tp_prev_frame >= min_tp_sequence_gap:
                tp_current_sequence = []
                tp_sequences.append(tp_current_sequence)
            tp_current_sequence.append(new_metadata)
            tp_prev_frame = frame



        # Add the image and annotations if the annotation is one we are interested in
        elif metadata.has_labels(tp_labels):
            new_annotations = []
            for ann in metadata.annotations:
                if ann.label in tp_labels:
                    new_annotations.append(ann)
            new_metadata = ImageMetadata(metadata.filename, metadata.filesize)
            new_metadata.annotations = new_annotations
            tp_counts += 1

            # frame = int(new_metadata.filename.split('_')[-1].replace(".jpg", ""))
            frame = int(re.search('(\d+)(?!.*\d)', new_metadata.filename).group(0))

            # Create new nested list if frame is part of new sequence
            if frame < tp_prev_frame:
                tn_prev_frame = -min_tp_sequence_gap
            if frame - tp_prev_frame >= min_tp_sequence_gap:
                tp_current_sequence = []
                tp_sequences.append(tp_current_sequence)
            tp_current_sequence.append(new_metadata)
            tp_prev_frame = frame

        # Add the image without annotations if the annotation is one of the true negative classes
        elif TN_PRESENT and metadata.has_labels(tn_labels):
            new_metadata = ImageMetadata(metadata.filename, metadata.filesize)
            if tn_counts % tn_only_every == 0:  # Set only_every to skip in cases where the number of TN is too high
                # frame = int(new_metadata.filename.split('_')[-1].replace(".jpg", ""))
                frame = int(re.search('(\d+)(?!.*\d)', new_metadata.filename).group(0))
                # Create new nested list if frame is part of new sequence
                if frame < tn_prev_frame:
                    tn_prev_frame = -min_tn_sequence_gap
                if frame - tn_prev_frame >= min_tn_sequence_gap:
                    tn_current_sequence = []
                    tn_sequences.append(tn_current_sequence)
                tn_current_sequence.append(new_metadata)
                tn_prev_frame = frame
            tn_counts += 1

        # Else add as a negative
        elif negative_sampling != 0:
            new_metadata = ImageMetadata(metadata.filename, metadata.filesize)
            negative_images.append(new_metadata)

    # TP and TN
    print(" - {} images with TP: {}, {} with TN: {}, other: {}".format(tp_counts, tp_labels, tn_counts, tn_labels,
                                                                       len(negative_images)))
    print(
        " - {} sequences with TP: {}, {} with TN: {}, other: {}".format(len(tp_sequences), tp_labels, len(tn_sequences),
                                                                        tn_labels, len(negative_images)))

    if TN_PRESENT:
        tn_train_metadata, tn_val_metadata = train_test_split(tn_sequences, test_size=val_split,
                                                              random_state=random_seed)
    print(f"- TP sequences {len(tp_sequences)}")
    print(f"- TN images {len(negative_images)}")
    tp_train_metadata, tp_val_metadata = train_test_split(tp_sequences, test_size=val_split, random_state=random_seed)

    print()
    print("Allocation of training and val images:")
    print(" - {} TP train images".format(str(tp_train_metadata).count(",") + 1))
    print(" - {} TP val images".format(str(tp_val_metadata).count(",") + 1))
    if TN_PRESENT:
        print(" - {} TN train images".format(str(tn_train_metadata).count(",") + 1))
        print(" - {} TN val images".format(str(tn_val_metadata).count(",") + 1))

    train_metadata = []
    if TN_PRESENT:
        train_metadata.extend(tn_train_metadata)
    train_metadata.extend(tp_train_metadata)

    val_metadata = []
    if TN_PRESENT:
        val_metadata.extend(tn_val_metadata)
    val_metadata.extend(tp_val_metadata)

    # Flattens val and training set
    train_metadata = [data for sequence in train_metadata for data in sequence]
    val_metadata = [data for sequence in val_metadata for data in sequence]

    # Negative sampling
    if negative_sampling != 0:
        # Distribution of true positive sequences
        TP_lens = [len(l) for l in tp_sequences]
        # Divide negatives into sets
        TN_len = len(negative_images)
        tn_sequences = []
        offset = 0
        while TN_len > 0:
            # Grab a TN sequence with random length chosen from the TP lengths
            sub_len = np.random.choice(TP_lens, 1)[0]
            sub_len = min(sub_len, TN_len)
            tn_sequences.append([negative_images[i] for i in range(offset, offset + sub_len)])
            TN_len -= sub_len
            offset += sub_len

        print(f"Negative sampling, mode {negative_sampling}")
        print(f" - TN sequences: {len(tn_sequences)}")
        # How many sequences to use
        if negative_sampling > 0:
            num_neg = min(int(len(tp_sequences) * negative_sampling), len(tn_sequences))
            neg_step = len(tn_sequences) / num_neg
            print(f" - {neg_step} selection step")
            idx = np.unique(np.round(np.arange(0, len(tn_sequences), neg_step)).astype(np.int32))
            print(f" - {len(idx)} TN sequences to take")
            tn_sequences = [tn_sequences[i] for i in idx]

        train_neg_metadata, val_neg_metadata = train_test_split(tn_sequences,
                                                                test_size=val_split,
                                                                random_state=random_seed)
        tn_train_image_count = 0
        tn_val_image_count = 0
        for seq in train_neg_metadata:
            train_metadata.extend(seq)
            tn_train_image_count += len(seq)
        for seq in val_neg_metadata:
            val_metadata.extend(seq)
            tn_val_image_count += len(seq)
        print(f" - {len(train_neg_metadata)} TN train sequences")
        print(f" - {len(val_neg_metadata)} TN val sequences")
        print(f" - {tn_train_image_count} TN train images")
        print(f" - {tn_val_image_count} TN val images")

    print("Number of images in training set: {}".format(len(train_metadata)))
    print("Number of images in validation set: {}".format(len(val_metadata)))

    # Sort metadata
    # train_metadata = sorted(train_metadata, key=lambda v: v.filename)
    # val_metadata = sorted(val_metadata, key=lambda v: v.filename)

    # Store in project
    train_project = Project()
    val_project = Project()
    for metadata in train_metadata:
        train_project.add_image(metadata)
    for metadata in val_metadata:
        val_project.add_image(metadata)

    # train_project.summary()
    # val_project.summary()

    return train_project, val_project
