"""

Functions to convert the CSV formats from the DeNet and Darknet models to CVAT

"""
import os
from datetime import datetime, timedelta
from glob import glob

from visage.project.annotation import RectangleAnnotation
from visage.project.image import ImageMetadata
from visage.project.project import Project


def combine_bbx(run_directory):
    bbx_filenames = sorted(glob(os.path.join(run_directory, "run*", "*.bbx")))
    bbx_lines = dict()
    for bbx_filename in bbx_filenames:
        print(bbx_filename)
        fn = os.path.basename(os.path.dirname(bbx_filename))
        date = fn[4:4 + 10]
        with open(bbx_filename, "r") as file:
            lines = file.readlines()[1:]
            if date not in bbx_lines:
                bbx_lines[date] = []
            bbx_lines[date].extend(lines)

    for date, lines in bbx_lines.items():
        with open(os.path.join(run_directory, f"{date}_combined.bbx"), "w") as combined:
            combined.write(f"{len(lines)}\n")
            combined.writelines(lines)


# TODO chase up torsten about saving glider UDP header
def convert_mlboxv2(csv_path: str,
                    remove_from_start: str = "/mnt/ssd",
                    bbx_base="",
                    glider_jpg_folder="",
                    scale_coef=(1,1)):
    project = Project()
    labels_set = set()

    # To deal with combined bbx where seq_ids start at 0
    max_seq_id = 0
    seq_offset = 0

    # create dict seq_num->time->filename, to resolve error in mlbox timestamp
    glider_jpg_dict = {}
    if len(glider_jpg_folder) > 0:
        glider_jpg_files = glob(os.path.join(glider_jpg_folder, "*.jpg"))
        for gj_file in glider_jpg_files:
            gj_filedata = gj_file.rsplit('/', 1)[-1].rsplit('.', 1)[0].split('_')
            glider_id = gj_filedata[0]
            time = datetime.strptime(gj_filedata[1] + gj_filedata[2] + "000", "%Y%m%d%H%M%S%f")
            seq_num = gj_filedata[3]

            if (glider_id, seq_num) not in glider_jpg_dict:
                glider_jpg_dict[(glider_id, seq_num)] = {}
            glider_jpg_dict[(glider_id, seq_num)][time] = gj_file

    print("Glider jpg dic created from %s with %d entries" % (glider_jpg_folder, len(glider_jpg_dict)))

    if csv_path.endswith(".bbx"):
        is_bbx = True
    else:
        is_bbx = False
    with open(csv_path, "r") as file:
        lines = file.readlines()
        if is_bbx:
            lines = lines[1:]
        for line in lines:
            line = line.replace("{", "").replace("}", "")
            parts = [part.strip() for part in line.split(',')]
            if is_bbx:
                fparts = parts[0].split('_')
                glider_id = fparts[1]
                starttime = datetime.strptime(fparts[2] + "000", "%Y%m%d%H%M%S%f")
                ms_offset = int(fparts[3])
                true_time = starttime + timedelta(milliseconds=ms_offset)
                seq_num = '0' + fparts[-1][:-4]  # remove '.jpg' and prepend '0' to match seq_num styles

                # print(f"{fparts[2]} - {starttime} - {ms_offset} - {true_time}")
                # filename = f"{bbx_base}/{fparts[1]}_{true_time.strftime('%Y%m%d_%H%M%S%f')[:-3]}_0{fparts[4]}"
                filename = None
                if len(glider_jpg_folder) > 0 and (glider_id, seq_num) in glider_jpg_dict:
                    for key in glider_jpg_dict[(glider_id, seq_num)]:
                        dt = abs((key - true_time).seconds)
                        if dt <= 1:
                            filename = glider_jpg_dict[(glider_id, seq_num)][key]
                            break

                if filename is None:
                    # print(f'error: did not find ({glider_id}, {seq_num}, {true_time}) in glider jpg dict')
                    continue

            else:
                filename = parts[0]

            metadata = ImageMetadata(filename.replace(remove_from_start, ""))
            if metadata.get_key() not in project.image_dict:
                project.add_image(metadata)
            else:
                metadata = project.image_dict[metadata.get_key()]

            if is_bbx:
                len_det_str = 9
                num_ann = (len(parts) - 1) // len_det_str
                for i in range(num_ann):
                    offset = i * len_det_str
                    v = parts[offset + 1:offset + len_det_str + 2]
                    label = v[0]
                    seq_id = int(v[2])
                    seq_idx = int(v[3])

                    # Max sequence id in this set
                    if seq_id == 0:
                        seq_offset += max_seq_id
                        print(f"offset is {seq_offset} (+{max_seq_id})")
                        max_seq_id = 0
                    if seq_id > max_seq_id:
                        # print(f"max_seq_id is ({max_seq_id})")
                        max_seq_id = seq_id
                    seq_id += seq_offset

                    # if seq_id not in latest_seq_idx:
                    #     latest_seq_idx[seq_id] = seq_idx
                    # elif latest_seq_idx[seq_id] == seq_idx:
                    #     continue
                    # latest_seq_idx[seq_id] = seq_idx

                    if v[4] == "" or v[4] is None:
                        label = label + "_NV"
                    elif v[4] == "False":
                        label = label + "_FP"

                    ann = RectangleAnnotation(x=int(v[5]),
                                              y=int(v[6]),
                                              width=int(v[7]),
                                              height=int(v[8]),
                                              label=label,
                                              score=float(v[1]),
                                              seq_id=seq_id,
                                              seq_idx=seq_idx)
                    labels_set.add(ann.label)
                    metadata.add_annotation(ann)
            else:
                len_det_str = 8
                num_ann = (len(parts) - 1) // len_det_str
                for i in range(num_ann):
                    offset = i * len_det_str
                    v = parts[offset + 1:offset + len_det_str + 2]

                    seq_id = int(v[2])
                    seq_idx = int(v[3])

                    # Max sequence id in this set
                    if seq_id == 0:
                        seq_offset += max_seq_id
                        print(f"offset is {seq_offset} (+{max_seq_id})")
                        max_seq_id = 0
                    if seq_id > max_seq_id:
                        # print(f"max_seq_id is ({max_seq_id})")
                        max_seq_id = seq_id
                    seq_id += seq_offset

                    # if seq_id not in latest_seq_idx:
                    #     latest_seq_idx[seq_id] = seq_idx
                    # elif latest_seq_idx[seq_id] == seq_idx:
                    #     continue
                    # latest_seq_idx[seq_id] = seq_idx

                    ann = RectangleAnnotation(x=int(v[4]),
                                              y=int(v[5]),
                                              width=int(v[6]),
                                              height=int(v[7]),
                                              label=v[0],
                                              score=float(v[1]),
                                              seq_id=seq_id,
                                              seq_idx=seq_idx)
                    labels_set.add(ann.label)
                    metadata.add_annotation(ann)
        project.add_labels(list(labels_set))

    # Clean up the trailing bounding boxes due to OF tracking staying active for 3+ seconds
    # e.g. the DeNet tracker will get one detection then have another 3 seconds (45!) worth of extra bounding boxes
    # 1. Work out what the maximum sequence index for each sequence is
    max_seq_idx = dict()
    for seq_id, annotations in project.sequence_dict().items():
        max_idx = 0
        for ann in annotations:
            if ann.seq_idx > max_idx:
                max_idx = ann.seq_idx
        max_seq_idx[seq_id] = max_idx
    # 2. Keep only 1 bounding box after the last bounding box with the max detection (e.g. the last detection)
    max_seq_idx_count = {key: 0 for key, _ in max_seq_idx.items()}
    th = 1
    for key, metadata in project.image_dict.items():
        filtered_annotations = []
        for ann in metadata.annotations:
            # Is it the bounding box from last detection?
            if ann.seq_idx == max_seq_idx[ann.seq_id]:
                max_seq_idx_count[ann.seq_id] += 1
                # Have we see less than threshold number of boxes? If so, keep it
                if max_seq_idx_count[ann.seq_id] <= th:
                    filtered_annotations.append(ann)
            # If not, keep it
            else:
                filtered_annotations.append(ann)
        # Replace with the filtered annotations
        metadata.annotations = filtered_annotations
    return project


def convert_mlboxv2_as_is(csv_path: str,
                          remove_from_start: str = "/mnt/ssd",
                          scale_coef=(1,1)):
    project = Project()
    labels_set = set()

    # To deal with combined bbx where seq_ids start at 0
    max_seq_id = 0
    seq_offset = 0

    # Is it a bbx file
    if csv_path.endswith(".bbx"):
        is_bbx = True
    else:
        is_bbx = False

    # Open the file and read all the lines
    with open(csv_path, "r") as file:
        lines = file.readlines()
        if is_bbx:
            lines = lines[1:]
        for line in lines:
            line = line.replace("{", "").replace("}", "")
            parts = [part.strip() for part in line.split(',')]

            # Filename
            filename = parts[0]
            if not os.path.isfile(filename):
                continue
            metadata = ImageMetadata(filename.replace(remove_from_start, ""))
            if metadata.get_key() not in project.image_dict:
                project.add_image(metadata)
            else:
                metadata = project.image_dict[metadata.get_key()]

            # BBX files have label and then "" | True | False for validation state
            if is_bbx:
                len_det_str = 9
                num_ann = (len(parts) - 1) // len_det_str
                for i in range(num_ann):
                    offset = i * len_det_str
                    v = parts[offset + 1:offset + len_det_str + 2]
                    label = v[0]
                    seq_id = int(v[2])
                    seq_idx = int(v[3])

                    # Max sequence id in this set
                    if seq_id == 0:
                        seq_offset += max_seq_id
                        print(f"offset is {seq_offset} (+{max_seq_id})")
                        max_seq_id = 0
                    if seq_id > max_seq_id:
                        # print(f"max_seq_id is ({max_seq_id})")
                        max_seq_id = seq_id
                    seq_id += seq_offset

                    if v[4] == "" or v[4] is None:
                        label = label + "_NV"
                    elif v[4] == "False":
                        label = label + "_FP"

                    ann = RectangleAnnotation(x=int(v[5] * scale_coef[0]),
                                              y=int(v[6] * scale_coef[1]),
                                              width=int(v[7] * scale_coef[0]),
                                              height=int(v[8] * scale_coef[1]),
                                              label=label,
                                              score=float(v[1]),
                                              seq_id=seq_id,
                                              seq_idx=seq_idx)
                    labels_set.add(ann.label)
                    metadata.add_annotation(ann)
            else:
                len_det_str = 8
                num_ann = (len(parts) - 1) // len_det_str
                for i in range(num_ann):
                    offset = i * len_det_str
                    v = parts[offset + 1:offset + len_det_str + 2]

                    seq_id = int(v[2])
                    seq_idx = int(v[3])

                    # Max sequence id in this set
                    if seq_id == 0:
                        seq_offset += max_seq_id
                        print(f"offset is {seq_offset} (+{max_seq_id})")
                        max_seq_id = 0
                    if seq_id > max_seq_id:
                        # print(f"max_seq_id is ({max_seq_id})")
                        max_seq_id = seq_id
                    seq_id += seq_offset

                    ann = RectangleAnnotation(x=int(float(v[4]) * scale_coef[0]),
                                              y=int(float(v[5]) * scale_coef[1]),
                                              width=int(float(v[6]) * scale_coef[0]),
                                              height=int(float(v[7]) * scale_coef[1]),
                                              label=v[0],
                                              score=float(v[1]),
                                              seq_id=seq_id,
                                              seq_idx=seq_idx)
                    labels_set.add(ann.label)
                    metadata.add_annotation(ann)
        project.add_labels(list(labels_set))

    # Clean up the trailing bounding boxes due to OF tracking staying active for 3+ seconds
    # e.g. the DeNet tracker will get one detection then have another 3 seconds (45!) worth of extra bounding boxes
    # 1. Work out what the maximum sequence index for each sequence is
    max_seq_idx = dict()
    for seq_id, annotations in project.sequence_dict().items():
        max_idx = 0
        for ann in annotations:
            if ann.seq_idx > max_idx:
                max_idx = ann.seq_idx
        max_seq_idx[seq_id] = max_idx
    # 2. Keep only 1 bounding box after the last bounding box with the max detection (e.g. the last detection)
    max_seq_idx_count = {key: 0 for key, _ in max_seq_idx.items()}
    th = 1
    for key, metadata in project.image_dict.items():
        filtered_annotations = []
        for ann in metadata.annotations:
            # Is it the bounding box from last detection?
            if ann.seq_idx == max_seq_idx[ann.seq_id]:
                max_seq_idx_count[ann.seq_id] += 1
                # Have we see less than threshold number of boxes? If so, keep it
                if max_seq_idx_count[ann.seq_id] <= th:
                    filtered_annotations.append(ann)
            # If not, keep it
            else:
                filtered_annotations.append(ann)
        # Replace with the filtered annotations
        metadata.annotations = filtered_annotations
    return project


# if __name__ == "__main__":
#     # Load project
#     projects = CVAT("http://cruncher-ph.nexus.csiro.au:8080")
#     projects.load()
#
#     # Get the project we are interested in
#     cvat_project = projects.load_project_by_name("Darknet Test Project")
#
#     # Load the darknet CSV data
#     csv = "/mnt/nas/datasets/heron_202007/runs/20200712_fitzroy_lagoon_to_east_2/darknet.csv"
#     project = convert_mlboxv2(csv)
#
#     # Create a task
#     filenames = [v.filename for k, v in project.image_dict.items()]
#     task_name = str(datetime.now())
#     cvat_project.create_task(task_name, filenames)
#
#     # Upload the annotations
#     cvat_project.create_annotations(task_name, create_task_annotations_patch(project, cvat_project.label_to_id_dict))


if __name__ == "__main__":
    combine_bbx("/mnt/nas/datasets/swains_202110_mlbox")

    # project = convert_mlboxv2("example_data/run_2021-09-28-11-56-48.bbx", bbx_base="asdfasdfasdfasdf")
    # project = convert_mlboxv2(
    #     '/mnt/nas/datasets/swains_202110_mlbox/run_2021-10-02-13-42-36/run_2021-10-02-13-42-36.bbx',
    #     glider_jpg_folder="/mnt/nas/datasets/swains_202110/20211002_reef_22-088a/jpg")
    # project.save_as("test.csv", format="mlboxv2")
