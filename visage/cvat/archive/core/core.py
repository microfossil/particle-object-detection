# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import json
import logging
import os
import requests
import pandas as pd
from io import BytesIO
import mimetypes
from time import sleep

from PIL import Image

from collections import OrderedDict

from .definition import ResourceType
log = logging.getLogger(__name__)


class CLI():

    def __init__(self, session, api, credentials):
        self.api = api
        self.session = session
        self.login(credentials)

    def tasks_data(self, task_id, resource_type, resources,chunk_size=0):
        """ Add local, remote, or shared files to an existing task. """
        url = self.api.tasks_id_data(task_id)
        data = {}
        files = None
        if resource_type == ResourceType.LOCAL:
            files = {'client_files[{}]'.format(i): open(f, 'rb') for i, f in enumerate(resources)}
        elif resource_type == ResourceType.REMOTE:
            data = {'remote_files[{}]'.format(i): f for i, f in enumerate(resources)}
        elif resource_type == ResourceType.SHARE:
            resources_list = list(resources)
            if len(resources_list) == 2 and resources_list[0] == 'file':
                df = pd.read_csv(resources_list[1], header=None)
                data = {'server_files[{}]'.format(i): f[0] for i, f in df.iterrows()}
            else:
                data = {'server_files[{}]'.format(i): f for i, f in enumerate(resources)}
        data['image_quality'] = 100
        data['use_cache'] = True
        data['copy_data'] = False
        if chunk_size != 0:
            data['chunk_size'] =  chunk_size

        response = self.session.post(url, data=data, files=files)
        response.raise_for_status()

    def projects_list(self, **kwargs):
        url = self.api.projects
        # print(url)
        response = self.session.get(url)
        response.raise_for_status()
        output = []

        projects = OrderedDict()
        response_json = response.json()
        for r in response_json['results']:
            # print(r)
            log.info('{id}, {name}, {status}'.format(**r))
            projects[r['name']] = r['id']
            # print(projects)

        return projects


    def tasks_list(self, use_json_output, **kwargs):
        """ List all tasks in either basic or JSON format. """
        url = self.api.tasks

        response = self.session.get(url)
        response.raise_for_status()
        output = OrderedDict()
        page = 1
        while True:
            response_json = response.json()
            for outp in response_json['results']:
                output[outp["name"]] = outp
                # output += response_json['results']
            for r in response_json['results']:
                if use_json_output:
                    log.info(json.dumps(r, indent=4))
                else:
                    log.info('{id},{name},{status}'.format(**r))
            if not response_json['next']:
                return output
            page += 1
            url = self.api.tasks_page(page)
            response = self.session.get(url)
            response.raise_for_status()
        return output


    def projects_create(self, name, labels, **kwargs):
        """ Create a new task with the given name and labels JSON
        """
        url = self.api.projects
        data = {'name': name,
                'labels': labels}
        response = self.session.post(url, json=data)
        response.raise_for_status()
        response_json = response.json()
        log.info('Created project ID: {id} NAME: {name}'.format(**response_json))
        project_id = response_json['id']
        return project_id

    def tasks_create(self, name, labels, overlap, chunk_size, segment_size, bug, resource_type, resources,
                     annotation_path='', annotation_format='CVAT XML 1.1',
                     completion_verification_period=20,
                     git_completion_verification_period=2,
                     dataset_repository_url='', project=None,
                     lfs=False, **kwargs):
        """ Create a new task with the given name and labels JSON and
        add the files to it. """
        url = self.api.tasks
        data = {'name': name,
                #'labels': labels,
                'overlap': overlap,
                'segment_size': segment_size,
                'bug_tracker': bug,
        }
        if project != None:
            print ('project', project, type(project))
            data.update({'project_id': project})
        else:
            data.update({'labels': labels})
        print (data)
        response = self.session.post(url, json=data)
        response.raise_for_status()
        response_json = response.json()
        log.info('Created task ID: {id} NAME: {name}'.format(**response_json))
        task_id = response_json['id']
        self.tasks_data(task_id, resource_type, resources,chunk_size)

        if annotation_path != '':
            url = self.api.tasks_id_status(task_id)
            response = self.session.get(url)
            response_json = response.json()

            log.info('Awaiting data compression before uploading annotations...')
            while response_json['state'] != 'Finished':
                sleep(completion_verification_period)
                response = self.session.get(url)
                response_json = response.json()
                logger_string= '''Awaiting compression for task {}.
                            Status={}, Message={}'''.format(task_id,
                                                            response_json['state'],
                                                            response_json['message'])
                log.info(logger_string)

            self.tasks_upload(task_id, annotation_format, annotation_path, **kwargs)
        if dataset_repository_url:
            response = self.session.post(
                        self.api.git_create(task_id),
                        json={
                            'path': dataset_repository_url,
                            'lfs': lfs,
                            'tid': task_id})
            response_json = response.json()
            rq_id = response_json['rq_id']
            log.info(f"Create RQ ID: {rq_id}")
            check_url = self.api.git_check(rq_id)
            response = self.session.get(check_url)
            response_json = response.json()
            log.info('''Awaiting dataset repository for task. Status: {}'''.format(
                    response_json['status']))
            while response_json['status'] != 'finished':
                sleep(git_completion_verification_period)
                response = self.session.get(check_url)
                response_json = response.json()
                if response_json['status'] == 'Failed':
                    log.error(f'Dataset repository creation request for task {task_id} failed.')

            log.info(f"Dataset repository creation completed with status: {response_json['status']}.")


    def delete_projects(self, project_ids, **kwargs):
        """ Delete a list of projects, ignoring those which don't exist. """
        print ('delete_projects')
        for project_id in project_ids:
            url = self.api.projects_id(project_id)
            response = self.session.delete(url)
            try:
                response.raise_for_status()
                log.info('Project ID {} deleted'.format(project_id))
            except requests.exceptions.HTTPError as e:
                if response.status_code == 404:
                    log.info('Project ID {} not found'.format(project_id))
                else:
                    raise e


    def tasks_delete(self, task_ids, **kwargs):
        """ Delete a list of tasks, ignoring those which don't exist. """
        for task_id in task_ids:
            url = self.api.tasks_id(task_id)
            response = self.session.delete(url)
            try:
                response.raise_for_status()
                log.info('Task ID {} deleted'.format(task_id))
            except requests.exceptions.HTTPError as e:
                if response.status_code == 404:
                    log.info('Task ID {} not found'.format(task_id))
                else:
                    raise e

    def tasks_frame(self, task_id, frame_ids, outdir='', quality='original', **kwargs):
        """ Download the requested frame numbers for a task and save images as
        task_<ID>_frame_<FRAME>.jpg."""
        for frame_id in frame_ids:
            url = self.api.tasks_id_frame_id(task_id, frame_id, quality)
            response = self.session.get(url)
            response.raise_for_status()
            im = Image.open(BytesIO(response.content))
            mime_type = im.get_format_mimetype() or 'image/jpg'
            im_ext = mimetypes.guess_extension(mime_type)
            # FIXME It is better to use meta information from the server
            # to determine the extension
            # replace '.jpe' or '.jpeg' with a more used '.jpg'
            if im_ext == '.jpe' or '.jpeg' or None:
                im_ext = '.jpg'

            outfile = 'task_{}_frame_{:06d}{}'.format(task_id, frame_id, im_ext)
            im.save(os.path.join(outdir, outfile))

    def tasks_dump(self, task_id, fileformat, filename, **kwargs):
        """ Download annotations for a task in the specified format
        (e.g. 'YOLO ZIP 1.0')."""
        url = self.api.tasks_id(task_id)
        response = self.session.get(url)
        response.raise_for_status()
        response_json = response.json()

        url = self.api.tasks_id_annotations_filename(task_id,
                                                     response_json['name'],
                                                     fileformat)
        while True:
            response = self.session.get(url)
            response.raise_for_status()
            log.info('STATUS {}'.format(response.status_code))
            if response.status_code == 201:
                break

        response = self.session.get(url + '&action=download')
        response.raise_for_status()

        with open(filename, 'wb') as fp:
            fp.write(response.content)

    def tasks_upload(self, task_id, fileformat, filename, **kwargs):
        """ Upload annotations for a task in the specified format
        (e.g. 'YOLO ZIP 1.0')."""
        url = self.api.tasks_id_annotations_format(task_id, fileformat)
        while True:
            response = self.session.put(
                url,
                files={'annotation_file': open(filename, 'rb')}
            )
            response.raise_for_status()
            if response.status_code == 201:
                break

        logger_string = "Upload job for Task ID {} ".format(task_id) +\
            "with annotation file {} finished".format(filename)
        log.info(logger_string)

    def login(self, credentials):
        url = self.api.login
        auth = {'username': credentials[0], 'password': credentials[1]}
        response = self.session.post(url, auth)
        response.raise_for_status()
        if 'csrftoken' in response.cookies:
            self.session.headers['X-CSRFToken'] = response.cookies['csrftoken']


class CVAT_API_V1():
    """ Build parameterized API URLs """

    def __init__(self, host, https=False):
        if host.startswith('https://'):
            https = True
        if host.startswith('http://') or host.startswith('https://'):
            host = host.replace('http://', '')
            host = host.replace('https://', '')
        scheme = 'https' if https else 'http'
        self.base = '{}://{}/api/v1/'.format(scheme, host)
        self.git = f'{scheme}://{host}/git/repository/'

    @property
    def tasks(self):
        return self.base + 'tasks'

    @property
    def projects(self):
        return self.base + 'projects'

    def tasks_page(self, page_id):
        return self.tasks + '?page={}'.format(page_id)

    def tasks_id(self, task_id):
        return self.tasks + '/{}'.format(task_id)

    def projects_id(self, project_id):
        return self.projects + '/{}'.format(project_id)

    def tasks_id_data(self, task_id):
        return self.tasks_id(task_id) + '/data'

    def tasks_id_frame_id(self, task_id, frame_id, quality):
        return self.tasks_id(task_id) + '/data?type=frame&number={}&quality={}'.format(frame_id, quality)

    def tasks_id_status(self, task_id):
        return self.tasks_id(task_id) + '/status'

    def tasks_id_annotations_format(self, task_id, fileformat):
        return self.tasks_id(task_id) + '/annotations?format={}' \
            .format(fileformat)

    def tasks_id_annotations_filename(self, task_id, name, fileformat):
        return self.tasks_id(task_id) + '/annotations?format={}&filename={}' \
            .format(fileformat, name)

    def git_create(self, task_id):
        return self.git + f'create/{task_id}'

    def git_check(self, rq_id):
        return self.git + f'check/{rq_id}'

    @property
    def login(self):
        return self.base + 'auth/login'
