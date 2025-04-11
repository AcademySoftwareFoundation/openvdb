#!/usr/bin/env python3
#
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# Python script to download the latest Houdini builds
# using the SideFX download API:
#
# https://www.sidefx.com/docs/api/download/index.html
#
# Authors: Dan Bailey, SideFX

import time
import sys
import re
import shutil
import json
import base64
import requests
import hashlib
import os
import argparse
import copy

# For progress bar printing
try:
    from tqdm import tqdm
    has_tqdm = True
except:
    has_tqdm = False
    pass

# Code that provides convenient Python wrappers to call into the API:

def service(
        access_token_url, client_id, client_secret_key, endpoint_url,
        access_token=None, access_token_expiry_time=None):
    if (access_token is None or
            access_token_expiry_time is None or
            access_token_expiry_time < time.time()):
        access_token, access_token_expiry_time = (
            get_access_token_and_expiry_time(
                access_token_url, client_id, client_secret_key))

    return _Service(
        endpoint_url, access_token, access_token_expiry_time)


class _Service(object):
    def __init__(
            self, endpoint_url, access_token, access_token_expiry_time):
        self.endpoint_url = endpoint_url
        self.access_token = access_token
        self.access_token_expiry_time = access_token_expiry_time

    def __getattr__(self, attr_name):
        return _APIFunction(attr_name, self)


class _APIFunction(object):
    def __init__(self, function_name, service):
        self.function_name = function_name
        self.service = service

    def __getattr__(self, attr_name):
        # This isn't actually an API function, but a family of them.  Append
        # the requested function name to our name.
        return _APIFunction(
            "{0}.{1}".format(self.function_name, attr_name), self.service)

    def __call__(self, *args, **kwargs):
        return call_api_with_access_token(
            self.service.endpoint_url, self.service.access_token,
            self.function_name, args, kwargs)

#---------------------------------------------------------------------------
# Code that implements authentication and raw calls into the API:


def get_access_token_and_expiry_time(
        access_token_url, client_id, client_secret_key):
    """Given an API client (id and secret key) that is allowed to make API
    calls, return an access token that can be used to make calls.
    """
    response = requests.post(
        access_token_url,
        headers={
            "Authorization": u"Basic {0}".format(
                base64.b64encode(
                    "{0}:{1}".format(
                        client_id, client_secret_key
                    ).encode()
                ).decode('utf-8')
            ),
        })
    if response.status_code != 200:
        raise AuthorizationError(response.status_code, response.text)

    response_json = response.json()
    access_token_expiry_time = time.time() - 2 + response_json["expires_in"]
    return response_json["access_token"], access_token_expiry_time


class AuthorizationError(Exception):
    """Raised from the client if the server generated an error while generating
    an access token.
    """
    def __init__(self, http_code, message):
        super(AuthorizationError, self).__init__(message)
        self.http_code = http_code


def call_api_with_access_token(
        endpoint_url, access_token, function_name, args, kwargs):
    """Call into the API using an access token that was returned by
    get_access_token.
    """
    response = requests.post(
        endpoint_url,
        headers={
            "Authorization": "Bearer " + access_token,
        },
        data=dict(
            json=json.dumps([function_name, args, kwargs]),
        ))
    if response.status_code == 200:
        return response.json()

    raise APIError(response.status_code, response.text)


class APIError(Exception):
    """Raised from the client if the server generated an error while calling
    into the API.
    """
    def __init__(self, http_code, message):
        super(APIError, self).__init__(message)
        self.http_code = http_code


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Download a Houdini Installation')
    parser.add_argument('version', type=str, help='Major.Minor version of Houdini to download')
    parser.add_argument('platform', type=str, help='Platform target')
    parser.add_argument('--prod', action='store_true', help='Only download production builds')
    parser.add_argument('--list', action='store_true', help='Just list the available builds and exit.')
    args = parser.parse_args()

    version = args.version
    platform = args.platform
    only_production = args.prod

    user_client_id = os.getenv('HOUDINI_CLIENT_ID')
    user_client_secret_key = os.getenv('HOUDINI_SECRET_KEY')

    if not re.match('[0-9][0-9]\.[0-9]$', version):
        raise IOError('Invalid Houdini Version "%s", expecting in the form "major.minor" such as "16.0"' % version)

    service = service(
            access_token_url="https://www.sidefx.com/oauth2/application_token",
            client_id=user_client_id,
            client_secret_key=user_client_secret_key,
            endpoint_url="https://www.sidefx.com/api/",
        )

    releases_list = service.download.get_daily_builds_list(
            product='houdini', version=version, platform=platform, only_production=only_production)

    print('Available builds:')
    for rel in releases_list:
        rel = copy.deepcopy(rel)
        if 'third_party_libraries' in rel:
            # Don't print these
            del rel['third_party_libraries']
        print(rel)

    if args.list:
        sys.exit(0)

    print('Selecting build: ' + releases_list[0]['build'])

    latest_release = service.download.get_daily_build_download(
            product='houdini', version=version, platform=platform, build=releases_list[0]['build'])
    print(latest_release)

    # Can't do this procedurally as latest_release['filename'] can contain
    # multiple periods and may have multiple trailing extensions...
    extension = ''
    if   'linux' in platform: extension = 'tar.gz'
    elif 'macos' in platform: extension = 'dmg'
    elif 'win64' in platform: extension = 'exe'
    assert(extension in latest_release['filename'])

    # Download the file and save it as hou.extension
    local_filename = 'hou.' + extension
    print('Writing to "' + local_filename + '"')

    response = requests.get(latest_release['download_url'], stream=True)
    if response.status_code == 200:
        response.raw.decode_content = True
        if has_tqdm:
            file_size = int(response.headers.get('Content-Length', 0))
            desc = "(Unknown total file size)" if file_size == 0 else ""
            with tqdm.wrapattr(response.raw, "read", total=file_size, desc=desc) as r_raw:
                with open(local_filename, 'wb') as f:
                    shutil.copyfileobj(r_raw, f)
        else:
            with open(local_filename, 'wb') as f:
                shutil.copyfileobj(response.raw, f)
    else:
        raise Exception('Error downloading file!')

    # Verify the file checksum is matching
    file_hash = hashlib.md5()
    with open(local_filename, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            file_hash.update(chunk)
    if file_hash.hexdigest() != latest_release['hash']:
        raise Exception('Checksum does not match!')
