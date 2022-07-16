from tqdm import tqdm
from google.cloud import storage
from collections import OrderedDict
from datetime import datetime
import json, sys
import pandas as pd
#sys.path.append('vertigo-processing')

from vertigo.via.project import ViaProject

def create_via_to_df(jsonfile, small):
    width = 2448
    height = 2048

    project = ViaProject().load(jsonfile)
    df_sequence = pd.DataFrame(columns=['frame', 'name', 'width', 'height', 'label', 'occluded', 'xtl', 'xbr', 'ybr'])
    for key, metadata in  tqdm(project.image_metadata_dict.items()):
        if small == 'True':
            key = key.replace('.jpg', '_small.jpg')
        if len(metadata.annotations) > 0:
            for ann in metadata.annotations:

                if small == 'True':


                    df_sequence = df_sequence.append(
                        {'frame': key, 'width': width / 2, 'height': height / 2,
                         'name': key,
                         'xtl': ann.x / 2,
                         'ytl': ann.y / 2,
                         'xbr': ann.x / 2 + ann.width / 2,
                         'ybr': ann.y / 2 + ann.height / 2,
                         'occluded': 0,
                         'label': ann.label
                         }, ignore_index=True)
                else:
                    df_sequence = df_sequence.append(
                        {'frame': key, 'width': width , 'height': height ,
                         'name': key,
                         'xtl': ann.x ,
                         'ytl': ann.y ,
                         'xbr': ann.x+ann.width ,
                         'ybr': ann.y+ann.height ,
                         'occluded': 0,
                         'label': ann.label
                         }, ignore_index=True)

    return df_sequence


def create_viaseq_to_df(jsonfile, small='False'):



    with open(jsonfile, 'r') as f:
        data = json.load(f, object_pairs_hook=OrderedDict)

    list_files = list(data['_via_img_metadata'].keys())
    df_sequence = pd.DataFrame(columns={'id_seq', 'fname', 'xtl', 'ytl', 'xbr', 'ybr', 'seq_len', 'label'})

    for nb_file in tqdm(range(len(list_files))):
        ff = list_files[nb_file]
        fname = data['_via_img_metadata'][ff]['filename'].split('/')[-1]
        if small == 'True':
            fname = fname.replace('.jpg', '_small.jpg')
        if len(data['_via_img_metadata'][ff]['regions']) > 0:
            for i in range(len(data['_via_img_metadata'][ff]['regions'])):
                annot = data['_via_img_metadata'][ff]['regions'][i]['shape_attributes']
                if small == 'True':
                    df_sequence = df_sequence.append(
                        {'id_seq': data['_via_img_metadata'][ff]['regions'][i]['region_attributes']['id'],
                         'fname': fname,
                         'xtl': annot['x'] / 2,
                         'ytl': annot['y'] / 2,
                         'xbr': annot['x'] / 2 + annot['width'] / 2,
                         'ybr': annot['y'] / 2 + annot['height'] / 2,
                         'seq_len': data['_via_img_metadata'][ff]['regions'][i]['region_attributes']['seq_len'],
                         'label': data['_via_img_metadata'][ff]['regions'][i]['region_attributes']['label']
                         }, ignore_index=True)
                else:
                    df_sequence = df_sequence.append(
                        {'id_seq': data['_via_img_metadata'][ff]['regions'][i]['region_attributes']['id'],
                         'fname': fname,
                         'xtl': annot['x'] ,
                         'ytl': annot['y'] ,
                         'xbr': annot['x']  + annot['width'] ,
                         'ybr': annot['y']  + annot['height'] ,
                         'seq_len': data['_via_img_metadata'][ff]['regions'][i]['region_attributes']['seq_len'],
                         'label': data['_via_img_metadata'][ff]['regions'][i]['region_attributes']['label']
                         }, ignore_index=True)
    return df_sequence


def create_xml_dumper(file_object, nb_files):

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

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )