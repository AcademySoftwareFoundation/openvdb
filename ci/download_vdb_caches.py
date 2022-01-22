#!/usr/bin/env python3

import os
import sys
import time
import threading
import zipfile
import argparse

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

parser = argparse.ArgumentParser(description='VDB files to download')
parser.add_argument('--files', '-f', type=str, nargs='+',
    help='files to download i.e. bunny.vdb fire.vdb')
args = parser.parse_args()

vdbs = [
    'armadillo.vdb',
    'boat_points.vdb',
    'buddha.vdb',
    'bunny.vdb',
    'bunny_cloud.vdb',
    'bunny_points.vdb',
    'crawler.vdb',
    'cube.vdb',
    'dragon.vdb',
    'emu.vdb',
    'explosion.vdb',
    'fire.vdb',
    'icosahedron.vdb',
    'iss.vdb',
    'smoke1.vdb',
    'smoke2.vdb',
    'space.vdb',
    'sphere.vdb',
    'sphere_points.vdb',
    'torus.vdb',
    'torus_knot.vdb',
    'utahteapot.vdb',
    'venusstatue.vdb',
    'waterfall_points.vdb'
]

if args.files:
    vdbs = list(set(vdbs) & set(args.files))
if not vdbs:
    raise RuntimeError('No valid VDBs specified for download')

# forma urls in the form:
#    'https://artifacts.aswf.io/io/aswf/openvdb/models/buddha.vdb/1.0.0/buddha.vdb-1.0.0.zip'

vdb_urls = []
for vdb in vdbs:
    url = 'https://artifacts.aswf.io/io/aswf/openvdb/models/' + \
        vdb +'/1.0.0/' + vdb + '-1.0.0.zip'
    vdb_urls.append(url)


def download(link, filelocation):
    urlretrieve(link, filelocation)

# Init downloads

downloads = dict()

for url in vdb_urls:
    zip_filename = os.path.basename(url)
    print('Initiating download "' + url + '"')
    download_thread = threading.Thread(target=download, args=(url,zip_filename))
    download_thread.start()
    downloads[zip_filename] = download_thread

sys.stdout.flush()

# Process files

while downloads:
    zip_file = None
    while not zip_file:
        time.sleep(1)
        for file, thread in downloads.items():
            if not thread.is_alive():
                thread.join()
                zip_file = file
                break
    # Remove the entry
    del downloads[zip_file]

    try:
        # Extract the downloaded zip
        print('Extracting "' + zip_file + '"...')
        sys.stdout.flush()
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall()

        print('Cleaning up "' + zip_file + '"...')
        sys.stdout.flush()
        if os.path.isfile(zip_file): os.remove(zip_file)

    except Exception as e:
        print(e)
        pass

