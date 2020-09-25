#!/usr/bin/env bash

set -ex

CUDA_VER="$1"; shift

#wget https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-repo-rhel7-9.2.148-1.x86_64.rpm
#rpm -i cuda-repo-rhel7-9.2.148-1.x86_64.rpm

sudo yum -y install cuda-${CUDA_VER}
