FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    build-essential \
    bzip2 \
    cmake \
    curl \
    git \
    g++ \
    pkg-config \
    rsync \
    software-properties-common \
    sudo \
    tar \
    timidity \
    unzip \
    wget \
    locales \
    zlib1g-dev \
    python3-dev \
    python3 \
    python3-pip \
    python3-tk \
    libjpeg-dev \
    libpng-dev

# Python3
RUN pip3 install pip --upgrade
RUN pip3 install cython timeout_decorator \
  numpy \
  matplotlib

RUN pip3 install scipy
RUN pip3 install Pillow
RUN pip3 install scikit-image
RUN pip3 install keras===2.1.6
RUN pip3 install opencv-python
RUN pip3 install h5py
RUN pip3 install imgaug
RUN pip3 install IPython[all]
RUN pip3 install tqdm==4.19.9
RUN pip3 install jupyter-client==5.3.4
RUN pip3 install jupyter-core==4.6.1
RUN pip3 install pandas
RUN pip3 install pickle-mixin

RUN pip3 install tensorflow==1.15.0
RUN pip3 install opencv-python

# Unicode support:
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Enables X11 sharing and creates user home directory
ENV USER_NAME CGIAR
ENV HOME_DIR /home/$USER_NAME
#
# Replace HOST_UID/HOST_GUID with your user / group id (needed for X11)
ENV HOST_UID 1000
ENV HOST_GID 1000

RUN export uid=${HOST_UID} gid=${HOST_GID} && \
    mkdir -p ${HOME_DIR} && \
    echo "$USER_NAME:x:${uid}:${gid}:$USER_NAME,,,:$HOME_DIR:/bin/bash" >> /etc/passwd && \
    echo "$USER_NAME:x:${uid}:" >> /etc/group && \
    echo "$USER_NAME ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/$USER_NAME && \
    chmod 0440 /etc/sudoers.d/$USER_NAME && \
    chown ${uid}:${gid} -R ${HOME_DIR}

USER ${USER_NAME}
WORKDIR ${HOME_DIR}
