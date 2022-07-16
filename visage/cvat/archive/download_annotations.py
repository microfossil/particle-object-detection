import os
import io
import shutil
import zipfile
import requests
from requests.auth import HTTPBasicAuth
from urllib.request import urlretrieve
from tqdm import tqdm
import time


def download_task_annotations(server, task_id, task_name, output_dir):
    types = [{"type": "CVAT%20for%20images%201.1", "filename": "image_annotations.xml"},
             {"type": "CVAT%20for%20video%201.1", "filename": "video_annotations.xml"}]
    for t in types:
        url = f"{server}/api/v1/tasks/{task_id}/annotations?format={t['type']}&action=download"
        data = io.BytesIO()
        retries = 0
        while retries < 5:
            req = requests.get(url, auth = HTTPBasicAuth('admin', 'admin'))
            if len(req.content) > 0:
                data = io.BytesIO(req.content)
                break
            else:
                time.sleep(1)
                retries += 1
        save_dir = f"{output_dir}/{task_name}"
        os.makedirs(save_dir, exist_ok=True)
        z = zipfile.ZipFile(data)
        z.extract(z.infolist()[0], save_dir)
        shutil.move(os.path.join(save_dir, "annotations.xml"), os.path.join(save_dir, t['filename']))


def download_project_annotations(server, project_id, output_dir):
    request_url = f"{server}/api/v1/projects/{project_id}/tasks"
    print(request_url)
    task_info = requests.get(request_url, auth=HTTPBasicAuth('admin', 'admin')).json()
    for result in task_info['results']:
        print('-' * 80)
        print(result['url'])
        print(result['id'])
        print(result['name'])
        download_task_annotations(server, result['id'], result['name'], output_dir)


if __name__ == "__main__":
    download_project_annotations("http://cruncher-ph.nexus.csiro.au:8080",
                                 12,
                                 "./test")


