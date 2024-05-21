ARG CUDA_VERSION=12.1.1
ARG CUDNN_VERSION=8

FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu20.04

ENV PATH /usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH /usr/lib64:/usr/local/cuda/lib64:/usr/local/cuda/lib:${LD_LIBRARY_PATH}

# # nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics

RUN echo "Acquire { https::Verify-Peer false }" > /etc/apt/apt.conf.d/99verify-peer.conf \
    && if [ -f /etc/apt/sources.list.d/cuda.list ]; then \
        rm /etc/apt/sources.list.d/cuda.list; \
    fi \
    && if [ -f /etc/apt/sources.list.d/nvidia-ml.list ]; then \
        rm /etc/apt/sources.list.d/nvidia-ml.list; \
    fi \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated ca-certificates \
    && rm /etc/apt/apt.conf.d/99verify-peer.conf \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
		wget \
		rsync \
		vim \
		git \
    	curl \
		ninja-build \
		cmake \
		build-essential \
		xauth \
		openssh-server \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

ENV PATH /opt/conda/bin:$PATH
ENV TORCH_CUDA_ARCH_LIST "6.1;7.0;7.5;8.0;8.6+PTX"

# used for cross-compilation in docker build
ENV FORCE_CUDA=1

WORKDIR /fvdb
COPY env/test_environment.yml .

RUN /opt/conda/bin/conda env create -f test_environment.yml \
    && /opt/conda/bin/conda clean -ya \
    && /opt/conda/bin/conda init bash
