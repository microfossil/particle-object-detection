import click, sys
from utils import upload_blob, create_xml_dumper, create_viaseq_to_df, create_via_to_df


import numpy as np
from collections import OrderedDict
from tqdm import tqdm
import pandas as pd
import os

google_credential = "./Vertigo3_Annotations-031f368ba73e.json"

def write_cvat_annotations(destrepo, df_cloud, df_sequence, df_sequence2, cvat_output):


    if cvat_output == '':
        cvat_output = "annotations.xml"
    output = open(destrepo + cvat_output, "w")
    dumper = create_xml_dumper(output, len(df_cloud))
    dumper.open_root()
    dumper.add_meta()
    if len(df_sequence) > 0:
        for id_seq in sorted(np.unique(df_sequence['id_seq'].values.tolist())):
            df = df_sequence[df_sequence.id_seq == id_seq]

            pos = 0
            for index, row in df.iterrows():
                outside = '0'
                if row['fname'] in df_cloud['filename'].tolist():
                    if pos == 0:
                        dumper.open_track({'id': str(int(id_seq)), 'label': df.label.values[0], 'source': 'manual'})
                    frame_id = df_cloud.index[df_cloud['filename'] == row['fname']][0]
                    dumper.open_box({'frame': str(frame_id),
                                     'outside': outside,
                                     'occluded': "0", 'keyframe': "1",
                                     'xtl': str(row['xtl']),
                                     'ytl': str(row['ytl']),
                                     'xbr': str(row['xbr']),
                                     'ybr': str(row['ybr']), 'z_order': "0"})

                    dumper.close_box()
                    pos += 1



            if pos > 0:
                outside = '1'
                if row['fname'] in df_cloud['filename'].tolist():
                    if pos == 0:
                        dumper.open_track({'id': str(int(id_seq)), 'label': df.label.values[0], 'source': 'manual'})
                    frame_id = df_cloud.index[df_cloud['filename'] == row['fname']][0]
                    if frame_id < len(df_cloud)-1:
                        frame_id += 1
                    dumper.open_box({'frame': str(frame_id),
                                     'outside': outside,
                                     'occluded': "0", 'keyframe': "1",
                                     'xtl': str(row['xtl']),
                                     'ytl': str(row['ytl']),
                                     'xbr': str(row['xbr']),
                                     'ybr': str(row['ybr']), 'z_order': "0"})

                    dumper.close_box()
                dumper.close_track()

    for key in df_cloud['filename'].values:

        if len(df_sequence2)>0:
            df = df_sequence2[df_sequence2.frame == key]

            pos = 0
            for index, row in df.iterrows():

                frame_id = df_cloud.index[df_cloud['filename'] == row['frame']][0]
                if pos == 0:
                    dumper.open_image(OrderedDict([
                        ("id", str(frame_id)),
                        ("name", key),
                        ("width", str(row.width)),
                        ("height", str(row.height))
                    ]))
                dump_data = OrderedDict([
                    ("label", row.label),
                    ("occluded", str(int(0))),
                    ("source", 'manual'),
                    ('xtl', str(row.xtl)),
                    ('ytl', str(row.ytl)),
                    ('xbr', str(row.xbr)),
                    ('ybr', str(row.ybr)),
                    ('z_order', str(0)),

                ])
                dumper.open_box(dump_data)
                dumper.close_box()
                pos += 1
            if pos > 0:
                dumper.close_image()

    dumper.close_root()
    output.close()



'''python convert_cvat_annotation_all.py --jsonfile test/20200718_heron_dive_morings_v03_btk.json --jsonfile_seq test/20200718_heron_dive_morings_eval_compared_url_v04_KA.json --project_name 20200718_heron_dive_morings --small True --destrepo test/ --savecloud True'''
@click.command()
@click.option('--jsonfile', type=str, default='', prompt='via json file',
              help='Via json file.')
@click.option('--jsonfile_seq', type=str, default='', prompt='via json seq file',
              help='Via json seq file.')
@click.option('--project_name', type=str, default='', prompt='project name',
              help='Cloud files')
@click.option('--small', type=str, default='True' , prompt='small images',
              help='Small images')
@click.option('--destrepo', prompt='destination repository',
              help='Destination repository', type=str, default="./")
@click.option('--savecloud', prompt='save in Google Cloud',
              help='save in Google Cloud',type=str, default="False")
def dump_as_cvat_annotation_seq(jsonfile, jsonfile_seq, destrepo, project_name,  small='True', savecloud='False'):
    """Simple program to convert via annotation fomat to cvat annotation format."""
    #jsonfile  ='test/20200718_heron_dive_morings_v03_btk.json'

    click.echo('Start Reading Via file ')
    click.echo('jsonfi'
               'le '+jsonfile)
    click.echo('jsonfile_seq '+jsonfile_seq)
    click.echo('destrepo ' + destrepo)
    click.echo('project_name ' + project_name)
    click.echo('small ' + small)
    click.echo('savecloud ' + savecloud)


    if small == 'True':
        vertigo_bucket = 'vertigo3_cloud_small'
    else:
        vertigo_bucket = 'vertigo3_cloud'

    df_sequence, df_sequence2 = pd.DataFrame(), pd.DataFrame()
    if jsonfile != '':
        df_sequence2 = create_via_to_df(jsonfile, small)
    if jsonfile_seq != '':
        df_sequence = create_viaseq_to_df(jsonfile_seq,small )


    click.echo('Start Conversion ')
    input, input1 = "file.tmp", "file1.tmp"
    os.system('gsutil ls  gs://' + vertigo_bucket + '/' + project_name + '/images > ' + input)
    nb = 0
    with open(input) as f:
        for repo in tqdm(f):
            repo1 = str(repo).strip()

            os.system('echo path > ' + input1)
            os.system('gsutil ls ' + repo1 + ' >> ' + input1)
            df_cloud = pd.read_csv(input1)
            cloudfile = project_name + '_' + str(nb) + '.csv'
            df_cloud['filename'] = [df_cloud.iloc[i].values[0].split('/')[-1] for i in range(len(df_cloud))]

            if small == 'False':
                cvat_output = project_name + '_all_' + str(nb) + '.xml'
            else:
                cvat_output = project_name + '_all_' + str(nb) + '_small.xml'

            write_cvat_annotations(destrepo, df_cloud, df_sequence,df_sequence2, cvat_output)

            # Save in google cloud
            if savecloud == 'True':

                os.environ[
                    "GOOGLE_APPLICATION_CREDENTIALS"] = google_credential
                upload_blob(vertigo_bucket, destrepo+cvat_output, 'Annotations/'+project_name+'/'+cvat_output)
                #upload_blob(vertigo_bucket,  jsonfile, 'Annotations/' + project_name + '/' + jsonfile.split('/')[-1])
                #upload_blob(vertigo_bucket, jsonfile_seq, 'Annotations/' + project_name + '/' + jsonfile_seq.split('/')[-1])

            nb += 1
    os.system('rm '+input+' '+input1)



if __name__ == '__main__':
    dump_as_cvat_annotation_seq()