import click

from visage.training.tf_configure_pipeline import create_tf2_pipeline
from visage.training.training_set import create_training_set_from_cvat_tasks


@click.group()
def cli():
    pass


@cli.command()
@click.option('--cvat_url', type=str,
              prompt='CVAT URL',
              help='URL of CVAT instance')
@click.option('--train_val', type=str, multiple=True,
              prompt='train / val tasks',
              help='Tasks to include in train / validation set as TASK_NAME@PROJECT_NAME')
@click.option('--test', type=str, multiple=True, default=None,
              help='Tasks to include in test set as TASK_NAME@PROJECT_NAME')
@click.option('--output_dir', type=str,
              prompt='output directory',
              help='Path to directory to save dataset')
@click.option('--tp_labels', default=None, multiple=True,
              help='Object classes for which annotations are included')
@click.option('--tn_labels', default=None, multiple=True,
              help='Object classes for which annotations are removed but images still included')
@click.option('--split', type=float, default=0.1,
              help='Proportion of images in validation set')
@click.option('--tn_tp_ratio', type=float, default=0,
              help='Ratio of TN to TP images, 0 = use all, > 0 use this ratio')
@click.option('--tfrecords/--no-tfrecords', default=True,
              prompt='create TF records',
              help='Create TF record files for each split')
@click.option('--save_images/--no-save_images', default=False,
              help='Save images to annotations folder')
@click.option('--resize_height', type=float, default=None,
              help='Resize images to maximum of this height')
@click.option('--as_greyscale/--no-as_greyscale', default=False,
              help='Convert the images to greyscale')
@click.option('--cvat_dir', type=str,
              prompt='CVAT base directory',
              help='Base directory of CVAT connected share')
def create_training_set(cvat_url,
                        train_val,
                        test,
                        output_dir,
                        tp_labels,
                        tn_labels,
                        split,
                        tn_tp_ratio,
                        tfrecords,
                        save_images,
                        resize_height,
                        as_greyscale,
                        cvat_dir):
    create_training_set_from_cvat_tasks(cvat_url,
                                        train_val=train_val,
                                        test=test,
                                        output_dir=output_dir,
                                        tp_labels=tp_labels,
                                        tn_labels=tn_labels,
                                        split=split,
                                        tn_tp_ratio=tn_tp_ratio,
                                        create_tf_records=tfrecords,
                                        save_images=save_images,
                                        resize_height=resize_height,
                                        as_greyscale=as_greyscale,
                                        local_dir=cvat_dir,
                                        cvat_dir="")


@cli.command()
@click.option('--path', type=str,
              prompt='path to parent directory of visage-ml-tf2objdet',
              help='Path to directory with the TF2 obj det directory structure')
@click.option('--dataset', type=str,
              prompt='name of dataset',
              help='Name of dataset to train (in annotations directory)')
@click.option('--model', type=str,
              prompt='name of model',
              help='Name of model that will be created')
@click.option('--pretrained_model', type=str,
              prompt='name of pretrained model',
              help='Name of pretrained model to base this off')
@click.option('--memory', type=int,
              prompt='memory requirements',
              help='Amount of memory in GB to request from HPC')
@click.option('--time', type=int,
              prompt='time for training',
              help='Number of days to perform training')
@click.option('--checkpoint', type=int, default=0,
              prompt='checkpoint number',
              help='Index of the checkpoint to load (0 for pretrained models from the zoo)')
@click.option('--input_size', type=int,
              prompt='max dimension of image',
              help='Maximum dimension of input image')
@click.option('--batch_size', type=int,
              prompt='batch size',
              help='Number of images in batch')
@click.option('--num_steps', type=int, default=None,
              help='number of training steps')
@click.option('--learning_rate', type=float, default=None,
              help='Base learning rate for training')
@click.option('--learning_rate_steps', type=int, default=None,
              help='Steps over which learning rate decays')
@click.option('--warmup_learning_rate', type=float, default=None,
              help='Initial learning rate which increases to base learning rate')
@click.option('--warmup_learning_rate_steps', type=int, default=None,
              help='Steps over which warmup learning rate increase to base learning rate')
@click.option('--device', type=str,
              help='Device we are training on [hpc, cruncher]')
@click.option('--gpus', type=int, default=2,
              help='Number of gpus to train on')
def create_pipeline(path,
                    dataset,
                    model,
                    pretrained_model,
                    memory,
                    time,
                    checkpoint,
                    input_size,
                    batch_size,
                    num_steps,
                    learning_rate,
                    learning_rate_steps,
                    warmup_learning_rate,
                    warmup_learning_rate_steps,
                    device,
                    gpus):
    create_tf2_pipeline(path,
                        dataset,
                        model,
                        pretrained_model,
                        memory,
                        time,
                        checkpoint,
                        input_size,
                        batch_size,
                        num_steps,
                        learning_rate,
                        learning_rate_steps,
                        warmup_learning_rate,
                        warmup_learning_rate_steps,
                        device,
                        gpus)


if __name__ == "__main__":
    cli()
