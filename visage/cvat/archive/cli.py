#!/usr/bin/env python3
#
# SPDX-License-Identifier: MIT
import logging
import requests
import sys
from http.client import HTTPConnection
from visage.cvat.archive.core.core import CLI, CVAT_API_V1
from visage.cvat.archive.core.definition import parser
log = logging.getLogger(__name__)


def config_log(level):
    log = logging.getLogger('core')
    log.addHandler(logging.StreamHandler(sys.stdout))
    log.setLevel(level)
    if level <= logging.DEBUG:
        HTTPConnection.debuglevel = 1


def main(args):
    actions = {'create': CLI.tasks_create,
               'create_project': CLI.projects_create,
               'delete': CLI.tasks_delete,
               'delete_projects': CLI.delete_projects,
               'ls': CLI.tasks_list,
                'ls_projects': CLI.projects_list,
               'frames': CLI.tasks_frame,
               'dump': CLI.tasks_dump,
               'upload': CLI.tasks_upload}
    if isinstance(args, str):
        args = args.split(" ")
    args = parser.parse_args(args)
    config_log(args.loglevel)
    with requests.Session() as session:
        api = CVAT_API_V1('%s:%s' % (args.server_host, args.server_port), args.https)
        cli = CLI(session, api, args.auth)
        try:
            output = actions[args.action](cli, **args.__dict__)
        except (requests.exceptions.HTTPError,
                requests.exceptions.ConnectionError,
                requests.exceptions.RequestException) as e:
            log.critical(e)
            output = "error"
        return output


if __name__ == '__main__':
    main(args[1:])
