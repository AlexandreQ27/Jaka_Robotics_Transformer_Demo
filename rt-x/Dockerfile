# syntax=docker/dockerfile:1.3

FROM python:3.9

RUN apt -y update
RUN apt install -y --fix-missing \
    build-essential \
    cmake \
    gfortran \
    git \
    wget \
    curl \
    pkg-config \
    python3-dev \
    python3-numpy \
    software-properties-common \
    zip \
    && apt clean && rm -rf /tmp/* /var/tmp/*

COPY . /RT-X
RUN --mount=type=cache,target=/root/.cache pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -U pip
RUN --mount=type=cache,target=/root/.cache pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r RT-X/requirements.txt
RUN --mount=type=cache,target=/root/.cache pip install -i https://pypi.tuna.tsinghua.edu.cn/simple jupyter
CMD jupyter notebook --ip 0.0.0.0 --allow-root --notebook-dir=$PWD

