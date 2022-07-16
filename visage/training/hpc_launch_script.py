"""
IMPORTANT: strings below have to have no leading spaces else the generated scripts will not work properly on the HPC
"""
def generate(dataset_name,
             model_name,
             time=5,
             memory=128,
             obj_det_dir="/scratch1/$IDENT/visage-ml-tf2objdet",
             tf_model_dir="/scratch1/$IDENT/tensorflow_models",
             python_env_dir="/scratch1/$IDENT/envs/obj_det",
             on_hpc=True,
             gpus=4,
             ):
    """
    Generates the training scripts for running on the HPC
    :param dataset_name: the name of the dataset
    :param model_name: the name of the model
    :param time: time in HPC format, e.g. '3-00:00:00' for three days
    :param memory: amount of memory to request, e.g. '32gb'
    :param on_hpc: create the script for running on the hpc
    :param gpus: the number of gpus to use
    :return:
    """
    if isinstance(time, int):
        time = f"{time}-00:00:00"

    CUDA_MODULE = "cuda/11.2.1"
    CUDNN_MODULE = "cudnn/8.1.1-cuda112"
    PROTOBUF_MODULE = "protobuf/3.15.8"
    PYTHON_MODULE = "python/3.9.4"

    if on_hpc:
        setup = \
f"""
module load {CUDA_MODULE}
module load {CUDNN_MODULE}
module load {PROTOBUF_MODULE}
module load {PYTHON_MODULE}

source {python_env_dir}/bin/activate
"""
    else:
        setup = \
f"""
export PYTHONPATH=$PYTHONPATH:{tf_model_dir}
"""

    train = \
f"""#!/bin/bash
#SBATCH --job-name=train_{dataset_name}_{model_name}
#SBATCH --qos=express
#SBATCH --time={time}
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:{gpus}
#SBATCH --mem={memory}gb

echo train_{dataset_name}_{model_name}

IDENT=$(whoami)

{setup}

# The model
DATASET={dataset_name}
MODEL={model_name}

# Directory to run this script from
DIR={obj_det_dir}

# Location of config and model
MODEL_DIR=$DIR/models/$DATASET/$MODEL
PIPELINE_CONFIG=$MODEL_DIR/pipeline.config

# Location of object detection API
OBJ_DET_DIR={tf_model_dir}/research/object_detection

nvidia-smi

# Start training
cd $DIR
python $OBJ_DET_DIR/model_main_tf2.py \
--pipeline_config_path=$PIPELINE_CONFIG \
--model_dir=$MODEL_DIR \
--checkpoint_every_n=200 \
--num_workers={gpus}
"""

    # ----------------------------------------------------------------------------------------------------------------------
    # Create evaluation script
    # ----------------------------------------------------------------------------------------------------------------------
    evaluate = \
f"""#!/bin/bash
#SBATCH --job-name=eval_{dataset_name}_{model_name}
#SBATCH --qos=express
#SBATCH --time={time}
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem={memory}gb

echo eval_{dataset_name}_{model_name}

IDENT=$(whoami)

{setup}

# The model
DATASET={dataset_name}
MODEL={model_name}

# Directory to run this script from
DIR={obj_det_dir}

# Location of config and model
MODEL_DIR=$DIR/models/$DATASET/$MODEL
PIPELINE_CONFIG=$MODEL_DIR/pipeline.config

# Location of object detection API
OBJ_DET_DIR={tf_model_dir}/research/object_detection

# Start training
cd $DIR
python $OBJ_DET_DIR/model_main_tf2.py \
--pipeline_config_path=$PIPELINE_CONFIG \
--model_dir=$MODEL_DIR \
--checkpoint_dir=$MODEL_DIR \
--alsologtostderror
"""

    test = \
f"""#!/bin/bash
#SBATCH --job-name=test_{dataset_name}_{model_name}
#SBATCH --qos=express
#SBATCH --time={time}
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem={memory}gb

echo test_{dataset_name}_{model_name}

IDENT=$(whoami)

{setup}

# The model
DATASET={dataset_name}
MODEL={model_name}

# Directory to run this script from
DIR={obj_det_dir}

# Location of config and model
MODEL_DIR=$DIR/models/$DATASET/$MODEL
PIPELINE_CONFIG=$MODEL_DIR/pipeline_test.config

# Location of object detection API
OBJ_DET_DIR={tf_model_dir}/research/object_detection

# Start training
cd $DIR
python $OBJ_DET_DIR/model_main_tf2.py \
--pipeline_config_path=$PIPELINE_CONFIG \
--model_dir=$MODEL_DIR \
--checkpoint_dir=$MODEL_DIR \
--alsologtostderror
"""

    # ----------------------------------------------------------------------------------------------------------------------
    # Create export script
    # ----------------------------------------------------------------------------------------------------------------------
    export = \
f"""#!/bin/bash
#SBATCH --job-name=exp_{dataset_name}_{model_name}
#SBATCH --qos=express
#SBATCH --time=00:10:00
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem={memory}gb

echo export_{dataset_name}_{model_name}

IDENT=$(whoami)

{setup}

# The model
DATASET={dataset_name}
MODEL={model_name}

# Directory to run this script from
DIR={obj_det_dir}

# Location of config and model
MODEL_DIR=$DIR/models/$DATASET/$MODEL
PIPELINE_CONFIG=$MODEL_DIR/pipeline.config

# Location of object detection API
OBJ_DET_DIR={tf_model_dir}/research/object_detection

cd $DIR
python $OBJ_DET_DIR/exporter_main_v2.py \
--input_type image_tensor \
--pipeline_config_path $MODEL_DIR/pipeline.config \
--trained_checkpoint_dir $MODEL_DIR \
--output_directory $DIR/exported-models/$DATASET/$MODEL
"""

    return train, evaluate, test, export