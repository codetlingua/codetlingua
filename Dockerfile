FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    openjdk-11-jdk \
    g++-11 \
    gcc-11 \
    clang-14 \
    golang-1.20 \
    git \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

ENV PATH=$PATH:/lib/go-1.20/bin

SHELL ["/bin/bash", "-c"]

WORKDIR /home

RUN git clone https://github.com/codetcombat/codetcombat.git
