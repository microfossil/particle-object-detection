import os
from collections import OrderedDict
from glob import glob
from tqdm import tqdm
from visage.project.project import Project
from pathlib import Path


def ls_dir(path):
    return sorted([d for d in glob(os.path.join(path, "*")) if os.path.isdir(d)])


def ls(path, ext="*"):
    dir = sorted(glob(os.path.join(path, ext)))
    if isinstance(dir, str):
        dir = [dir]
    return dir


def combine_projects(base_dir, ext="*.json"):
    # Get latest projects in director
    project_files = latest_projects(base_dir, ext)
    # New project to save
    main_project = Project()
    # Add all projects
    for project_file in tqdm(project_files):
        main_project.add_project(Project.load(project_file))
    return main_project


def latest_projects(base_dir, ext="*.json"):
    # All the project folders
    dirs = ls_dir(base_dir)
    # List
    projects = dict()
    for dir in dirs:
        # Latest project in folder
        project = ls(dir, ext=ext)[-1]
        # Add to list
        projects[str(Path(dir).stem)] = project
    return projects


def latest_project(base_dir, ext="*.json"):
    # Latest project in folder
    return ls(base_dir, ext=ext)[-1]


def create_cvat_xml_dumper(file_object, nb_files):
    from xml.sax.saxutils import XMLGenerator
    class XmlAnnotationWriter:
        def __init__(self, file):
            self.version = "1.1"
            self.file = file
            self.xmlgen = XMLGenerator(self.file, 'utf-8')
            self._level = 0

        def _indent(self, newline = True):
            if newline:
                self.xmlgen.ignorableWhitespace("\n")
            self.xmlgen.ignorableWhitespace("  " * self._level)

        def _add_version(self):
            self._indent()
            self.xmlgen.startElement("version", {})
            self.xmlgen.characters(self.version)
            self.xmlgen.endElement("version")

        def open_root(self):
            self.xmlgen.startDocument()
            self.xmlgen.startElement("annotations", {})
            self._level += 1
            self._add_version()

        def _add_meta(self, meta):
            self._level += 1
            for k, v in meta.items():

                if isinstance(v, OrderedDict):
                    self._indent()
                    self.xmlgen.startElement(k, {})
                    self._add_meta(v)
                    self._indent()
                    self.xmlgen.endElement(k)
                elif isinstance(v, list):
                    self._indent()
                    self.xmlgen.startElement(k, {})
                    for tup in v:
                        self._add_meta(OrderedDict([tup]))
                    self._indent()
                    self.xmlgen.endElement(k)
                else:
                    self._indent()
                    self.xmlgen.startElement(k, {})
                    self.xmlgen.characters(v)
                    self.xmlgen.endElement(k)
            self._level -= 1

        def add_meta(self):
            self._indent()
            self.xmlgen.startElement("meta", {})
            newdicto = OrderedDict()
            newdicto['task']=OrderedDict()
            newdicto['task']['id']='1'
            newdicto['task']['name'] = "conversion"
            now = datetime.now()
            newdicto['task']['updated'] = now.strftime("%Y-%m-%d %H:%M:%S")
            newdicto['task']['segments']=OrderedDict()
            newdicto['task']['segments']['segment']=OrderedDict()
            newdicto['task']['segments']['segment']['id']='1'
            newdicto['task']['segments']['segment']['start']='0'
            newdicto['task']['segments']['segment']['stop'] = str(nb_files)

            newdicto['dumped'] = now.strftime("%Y-%m-%d %H:%M:%S")
            self._add_meta(newdicto)
            self._indent()
            self.xmlgen.endElement("meta")

        def open_track(self, track):
            self._indent()
            self.xmlgen.startElement("track", track)
            self._level += 1

        def open_image(self, image):
            self._indent()
            self.xmlgen.startElement("image", image)
            self._level += 1

        def open_box(self, box):
            self._indent()
            self.xmlgen.startElement("box", box)
            self._level += 1

        def open_polygon(self, polygon):
            self._indent()
            self.xmlgen.startElement("polygon", polygon)
            self._level += 1

        def open_polyline(self, polyline):
            self._indent()
            self.xmlgen.startElement("polyline", polyline)
            self._level += 1

        def open_points(self, points):
            self._indent()
            self.xmlgen.startElement("points", points)
            self._level += 1

        def open_cuboid(self, cuboid):
            self._indent()
            self.xmlgen.startElement("cuboid", cuboid)
            self._level += 1

        def open_tag(self, tag):
            self._indent()
            self.xmlgen.startElement("tag", tag)
            self._level += 1

        def add_attribute(self, attribute):
            self._indent()
            self.xmlgen.startElement("attribute", {"name": attribute["name"]})
            self.xmlgen.characters(attribute["value"])
            self.xmlgen.endElement("attribute")

        def close_box(self):
            self._level -= 1
            self._indent()
            self.xmlgen.endElement("box")

        def close_polygon(self):
            self._level -= 1
            self._indent()
            self.xmlgen.endElement("polygon")

        def close_polyline(self):
            self._level -= 1
            self._indent()
            self.xmlgen.endElement("polyline")

        def close_points(self):
            self._level -= 1
            self._indent()
            self.xmlgen.endElement("points")

        def close_cuboid(self):
            self._level -= 1
            self._indent()
            self.xmlgen.endElement("cuboid")

        def close_tag(self):
            self._level -= 1
            self._indent()
            self.xmlgen.endElement("tag")

        def close_image(self):
            self._level -= 1
            self._indent()
            self.xmlgen.endElement("image")

        def close_track(self):
            self._level -= 1
            self._indent()
            self.xmlgen.endElement("track")

        def close_root(self):
            self._level -= 1
            self._indent()
            self.xmlgen.endElement("annotations")
            self.xmlgen.endDocument()

    return XmlAnnotationWriter(file_object)