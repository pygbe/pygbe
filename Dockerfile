# Dockerfile for PyGBe
# --------------------
# To build the image:
# `nvidia-docker build --tag=pygbe:master .`
# To run a container:
# `nvidia-docker run --name=pygbe -it pygbe:master /bin/bash`
# If running a newer version of nvdidia-docker to run container
# `docker run --runtime=nvidia --name=pygbe -it pygbe:master /bin/bash`
# To access the software:
# Once in the container, pygbe can be found in `/opt/pygbe/pygbe-master`
# To stop the container:
# `nvidia-docker stop pygbe`
# To restart the container:
# `nvidia-docker restart pygbe`
# To access the container once you exited
# `nvidia-docker exec -it pygbe /bin/bash`
# To delete the container:
# `docker rm pygbe`


FROM nvidia/cuda:8.0-devel-ubuntu16.04

# Install basic requirements.
RUN apt-get update && \
    apt-get install -y wget unzip vim tmux  

# Install Miniconda.
RUN FILENAME=Miniconda3-4.3.21-Linux-x86_64.sh && \
    wget https://repo.continuum.io/miniconda/${FILENAME} -P /tmp && \
    bash /tmp/${FILENAME} -b -p /opt/miniconda && \
    export PATH=/opt/miniconda/bin:$PATH && \
    rm -f /tmp/${FILENAME}

# Add Miniconda to PATH.
ENV PATH=/opt/miniconda/bin:${PATH}

# Install required packages.
RUN conda install -y numpy=1.13.1 && \
    conda install -y scipy=0.19.1=np113py36_0 && \
    conda install -y matplotlib=2.0.2=np113py36_0 && \
    conda install -y swig=3.0.10 && \
    conda install -y requests=2.14.2 && \
    conda install -y pytest=3.5.1 && \
    pip install clint==0.5.1


# Install PyCUDA.
RUN VERSION=2017.1.1 && \
    TARBALL=pycuda-${VERSION}.tar.gz && \
    wget https://pypi.python.org/packages/b3/30/9e1c0a4c10e90b4c59ca7aa3c518e96f37aabcac73ffe6b5d9658f6ef843/${TARBALL} -P /tmp && \
    PYCUDA_DIR=/opt/pycuda/${VERSION} && \
    mkdir -p ${PYCUDA_DIR} && \
    tar -xzf /tmp/${TARBALL} -C ${PYCUDA_DIR} --strip-components=1 && \
    rm -f /tmp/${TARBALL} && \
    cd ${PYCUDA_DIR} && \
    python configure.py --cuda-root=/usr/local/cuda-8.0 && \
    make -j"$(nproc)" && \
    make install

# Install PyGBe
RUN wget https://github.com/barbagroup/pygbe/archive/master.zip -P /tmp && \
    mkdir -p /opt/pygbe && \
    unzip /tmp/master.zip -d /opt/pygbe/ && \
    rm -f /tmp/master.zip && \
    cd /opt/pygbe/pygbe-master && \
    python setup.py install clean
