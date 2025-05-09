FROM condaforge/miniforge3:24.11.3-2

ARG MODE=dev
RUN echo "Building fVDB container in $MODE mode"

# used for cross-compilation in docker build
ENV FORCE_CUDA=1

WORKDIR /openvdb

# force this CUDA version to be used to build the docker container because `docker build` does not
# expose the GPU to the docker build process for it to be detected
ENV CONDA_OVERRIDE_CUDA=12.0
# copy env/dev_environment.yml to /tmp/
COPY fvdb/env/dev_environment.yml /tmp/
RUN  conda env create -f /tmp/dev_environment.yml

RUN conda init
RUN echo "conda activate fvdb" >> ~/.bashrc
