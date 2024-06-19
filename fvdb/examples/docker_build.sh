#!/bin/bash
# Run by
#
# ./build.sh $GIT_ACCESS_TOKEN
#
# To get the access token, please visit
# https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token
#
GIT_ACCESS_TOKEN=$1
TAG="${2:-latest}"

set -x

docker build --build-arg GIT_ACCESS_TOKEN=$GIT_ACCESS_TOKEN -t fvdb:$TAG .
