FROM nvcr.io/nvidia/pytorch:24.05-py3

ARG MODE=production
RUN echo "Building fVDB container in $MODE mode"

# used for cross-compilation in docker build
ENV FORCE_CUDA=1

WORKDIR /fvdb
COPY . .
RUN  pip install --no-cache-dir -r env/build_requirements.txt

RUN if [ "$MODE" = "production" ]; then \
     MAX_JOBS=$(free -g | awk '/^Mem:/{jobs=int($4/2.5); if(jobs<1) jobs=1; print jobs}')  \
     TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6+PTX" \
     python setup.py install; \
    fi