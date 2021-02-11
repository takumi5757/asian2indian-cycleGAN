FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update && \
    apt-get install -y \
    mesa-utils \
    cmake \
    build-essential

RUN pip install -U pip

COPY requirements.txt .

RUN pip install -r requirements.txt