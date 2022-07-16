import os
import click

from visage.project.project import Project


@click.group()
def cli():
    pass


@cli.command()
@click.option('--input_dir', type=str,
              prompt='input directory',
              help='Path to directory images')
@click.option('--output_dir', type=str,
              prompt='output directory',
              help='Path to save project')
@click.option('--name', type=str,
              prompt='project name',
              help='Name for the project file (without extension)')
def from_directory(input_dir, output_dir, name):
    project = Project.from_directory(input_dir)
    project.metadata["name"] = name
    project.save_as(os.path.join(output_dir, name + ".json"))
    project.save_as(os.path.join(output_dir, name + "_via.json"), "via")
    project.save_as(os.path.join(output_dir, name + ".xml"), "cvat")


if __name__ == "__main__":
    cli()