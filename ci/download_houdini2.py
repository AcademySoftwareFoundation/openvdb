#!/usr/local/bin/python
#
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0
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

# this argument is for the major.minor version of Houdini to download (such as 15.0, 15.5, 16.0)
version = sys.argv[1]
only_production = True if sys.argv[2] == 'ON' else False
user_client_id = sys.argv[3]
user_client_secret_key = sys.argv[4]

if not re.match('[0-9][0-9]\.[0-9]$', version):
    raise IOError('Invalid Houdini Version "%s", expecting in the form "major.minor" such as "16.0"' % version)


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
        raise AuthorizationError(response.status_code, reponse.text)

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


service = service(
        access_token_url="https://www.sidefx.com/oauth2/application_token",
        client_id=user_client_id,
        client_secret_key=user_client_secret_key,
        endpoint_url="https://www.sidefx.com/api/",
    )

releases_list = service.download.get_daily_builds_list(
        product='houdini', version=version, platform='linux', only_production=only_production)

latest_release = service.download.get_daily_build_download(
        product='houdini', version=version, platform='linux', build=releases_list[0]['build'])

# Download the file as hou.tar.gz
local_filename = 'hou.tar.gz'
response = requests.get(latest_release['download_url'], stream=True)
if response.status_code == 200:
    with open(local_filename, 'wb') as f:
        response.raw.decode_content = True
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
