import os

from visage.training.hpc_launch_script import generate
from visage.training.tf_config import TFConfig


def quote(s):
    return "\"" + s + "\""


def create_tf2_pipeline(path,
                        dataset,
                        model,
                        pretrained_model,
                        memory=128,
                        time=5,
                        checkpoint=0,
                        input_size=1024,
                        batch_size=16,
                        num_steps=None,
                        learning_rate=None,
                        total_steps=None,
                        warmup_learning_rate=None,
                        warmup_steps=None,
                        device=None,
                        gpus=2
                        ):
    """
    Creates a Tensorflow object detection pipeline
    :param path: Path to the visage-ml-objdet directory
    :param dataset: Name of the dataset to train (must exist in visage-ml-objdet/annotations/DATASET_NAME)
    :param model: Name of the model to train (this will be created)
    :param pretrained_model: Name of the pretrained model to use as a base
    :param input_size: Input size of the images (used for both width and height)
    :param memory: Memory in GB for the HPC
    :param time: Time in days on the HPC
    :param checkpoint: The checkpoint to use (0 for pretrained model from TF2 zoo, else whatever the latest was)
    :param batch_size: Number of images in a batch (try 8 for Faster RCNN, 4 for EfficientDet)
    :param num_steps: Number of steps (batches) to run the training
    :return:
    """
    print(f"record path {path}")
    print(device)
    on_hpc = False
    # Directories of this code
    if device == "hpc":
        on_hpc = True
        gpus = 4
        obj_det_dir = "/scratch1/$IDENT/visage-ml-tf2objdet"
        tf_model_dir = "/scratch1/$IDENT/tensorflow_models"
        python_env_dir = "/scratch1/$IDENT/envs/obj_det"
    elif device == "cruncher":
        HOME = os.path.expanduser("~")
        gpus = 2
        obj_det_dir = HOME + "/code/visage-ml-tf2objdet"
        tf_model_dir = HOME + "/code/tensorflow_models"
        python_env_dir = HOME + "/venvs/tf2"
    elif device == "tactile":
        obj_det_dir = "~/Development/Visage/visage-ml-tf2objdet"
        tf_model_dir = "~/Development/tensorflow_models"
        python_env_dir = ""
    else:
        obj_det_dir = path
        tf_model_dir = os.path.join(path, "..", "tensorflow_models")
        python_env_dir = os.path.join(path, "..", "..", "venvs", "tf2")

    # Using convention to locate files
    labelmap = os.path.join("annotations", dataset, "label_map.txt")
    train_tfrecord = os.path.join("annotations", dataset, "train.tfrecord")
    val_tfrecord = os.path.join("annotations", dataset, "val.tfrecord")
    test_tfrecord = os.path.join("annotations", dataset, "test.tfrecord")

    # Locations to save files
    output_dir = os.path.join(obj_det_dir, "models", dataset, model)
    launch_output_dir = os.path.join(obj_det_dir, "launch", dataset, model)
    print(dataset)
    print(obj_det_dir)
    print(output_dir)
    print(launch_output_dir)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(launch_output_dir, exist_ok=True)

    # Get number of classes from
    with open(os.path.join(obj_det_dir, labelmap)) as f:
        num_classes = f.read().count("item {")

    # Load original config
    # - check in normal directory
    if os.path.exists(os.path.join(obj_det_dir, pretrained_model, "pipeline.config")):
        pass
    elif os.path.exists(os.path.join(obj_det_dir, "pre-trained-models",  pretrained_model, "pipeline.config")):
        pretrained_model = os.path.join("pre-trained-models", pretrained_model)
    elif os.path.exists(os.path.join(obj_det_dir, "exported-models",  pretrained_model, "pipeline.config")):
        pretrained_model = os.path.join("exported-models", pretrained_model)
    else:
        raise ValueError(f"Could not find the model {pretrained_model}")
    config = TFConfig.load(os.path.join(obj_det_dir, pretrained_model, "pipeline.config"))

    # Image resizer - SSD needs special treatment
    if "ssd" in config.data.model:
        # Resize and pad to make square
        resizer = {"keep_aspect_ratio_resizer": {"min_dimension": input_size,
                                                 "max_dimension": input_size,
                                                 "pad_to_max_dimension": True}}

        # # Add hard example miner
        # config.data.model.ssd.loss.hard_example_miner = DottedDict()
        # config.data.model.ssd.loss.hard_example_miner.num_hard_examples = 64
        # config.data.model.ssd.loss.hard_example_miner.iou_threshold = 0.7
        # config.data.model.ssd.loss.hard_example_miner.loss_type = "CLASSIFICATION"
        # config.data.model.ssd.loss.hard_example_miner.max_negatives_per_positive = 3
        # config.data.model.ssd.loss.hard_example_miner.min_negatives_per_image = 1
        #
        # # Change classification loss
        # config.data.model.ssd.loss.classification_loss = {"weighted_softmax": {}}
    else:
        # Resize keeping aspect ratio
        resizer = {"keep_aspect_ratio_resizer": {"min_dimension": input_size, "max_dimension": input_size}}

    # Learning
    if num_steps is not None:
        config.set("num_steps", num_steps)
    if learning_rate is not None:
        # Learning rate
        config.set("learning_rate_base", learning_rate)
    if total_steps is not None:
        # Total steps
        config.set("total_steps", total_steps)
    if warmup_learning_rate is not None:
        # Initial learning rate
        config.set("warmup_learning_rate", warmup_learning_rate)
    if warmup_steps is not None:
        # Warmup steps
        config.set("warmup_steps", warmup_steps)

    # Augmentations
    if pretrained_model.startswith("centernet"):  # CENTRENET DOES NOT WORK STILL
        random_horizontal_flip = {
            "random_horizontal_flip": {
                "keypoint_flip_permutation": [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
            }
        }
        random_vertical_flip = {
            "random_horizontal_flip": {
                "keypoint_flip_permutation": [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
            }
        }
    else:
        random_horizontal_flip = {"random_horizontal_flip": {}}
        random_vertical_flip = {"random_vertical_flip": {}}

    augmentations = [
        random_horizontal_flip,
        random_vertical_flip,
        {"random_adjust_brightness": {}},
        {"random_adjust_contrast": {}},
        {"random_adjust_saturation": {}},
        {"random_jitter_boxes": {"ratio": 0.05}},
        # Removed random crop as it seems to need one bounding box...
        # {"random_crop_pad_image":
        #      {"min_object_covered": 1.0,
        #       "min_aspect_ratio": 0.75,
        #       "max_aspect_ratio": 1.33,
        #       "min_area": 0.1,
        #       "max_area": 1.0,
        #       "overlap_thresh": 0.3,
        #       "clip_boxes": True,
        #       "random_coef": 0.1}}
    ]

    # ------------------------------------------------------------------------------------------------------------------
    # PIPELINE.CONFIG
    # ------------------------------------------------------------------------------------------------------------------
    # This is some helper code to automatically create a pipeline.config file for model training
    # pipeline.config files are in protobuf format and have a weird syntax that is almost like json but not quite
    # 1. key, value pairs are the same: {"key": value} but the keys dont have to have quotes
    # 2. repeated dictionary elements do not use [ ], instead they simply repeat, e.g. { "item": {}, "item": {}}
    #
    # 3. lists of values use the  [ ] format. e.g. {"key": [1, 2, 3] }
    # however, in the below code please enclose the list in quotes for it to work
    #
    # 4. some values require quotes, e.g. path strings, some do not. Check the pre-trained model pipeline.config.
    # use the helper method quote() to make this easier
    #
    # The TFConfig object can help with creating the pipeline.config class
    # - the set method finds the first instance of a key (at any depth) and replaces its value
    # - config.data provides access to the pipeline.config in dot separated format.
    # - remember to enclose lists of values in quotes, e.g. "[1, 2, 3]" (the quotes will not appear in the final file)
    # - if a value is supposed to be surrounded by quotes, use the qoute helper method,
    #   e.g. quote("path_to_to_training_file")

    # The set function will find the first instead
    config.set("num_classes", num_classes)
    config.set("image_resizer", resizer)
    config.set("use_static_shapes", False)

    config.data.train_config.batch_size = batch_size
    config.data.train_config.fine_tune_checkpoint = quote(
        os.path.join(pretrained_model, "checkpoint", "ckpt-{}".format(checkpoint)))
    config.data.train_config.fine_tune_checkpoint_type = quote("detection")
    config.data.train_config.data_augmentation_options = augmentations

    config.data.train_input_reader.label_map_path = quote(labelmap)
    config.data.train_input_reader.tf_record_input_reader.input_path = quote(train_tfrecord)
    config.data.eval_input_reader.label_map_path = quote(labelmap)
    config.data.eval_input_reader.tf_record_input_reader.input_path = quote(val_tfrecord)

    # config.data.model.faster_rcnn.first_stage_anchor_generator = {
    #       "grid_anchor_generator": {
    #         "scales": "[1.0, 2.0]",
    #         "aspect_ratios": "[0.5, 1.0, 2.0]",
    #         "height_stride": 16,
    #         "width_stride": 16,
    #       }
    #     }

    config.save_as(os.path.join(output_dir, "pipeline.config"))

    config.data.eval_input_reader.label_map_path = quote(labelmap)
    config.data.eval_input_reader.tf_record_input_reader.input_path = quote(test_tfrecord)

    config.save_as(os.path.join(output_dir, "pipeline_test.config"))

    # ------------------------------------------------------------------------------------------------------------------
    # LAUNCH SCRIPTS
    # ------------------------------------------------------------------------------------------------------------------
    train, eval, test, export = generate(dataset,
                                         model,
                                         time,
                                         memory,
                                         obj_det_dir,
                                         tf_model_dir,
                                         python_env_dir,
                                         on_hpc,
                                         gpus)
    with open(os.path.join(launch_output_dir, "train.sh"), "w") as f:
        f.write(train)
    with open(os.path.join(launch_output_dir, "eval.sh"), "w") as f:
        f.write(eval)
    with open(os.path.join(launch_output_dir, "test.sh"), "w") as f:
        f.write(test)
    with open(os.path.join(launch_output_dir, "export.sh"), "w") as f:
        f.write(export)
