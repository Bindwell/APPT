FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

LABEL authors="Colby T. Ford <colby@tuple.xyz>"

## Install system requirements
RUN apt update && \
    apt-get install -y --reinstall \
        ca-certificates && \
    apt install -y \
        git \
        wget \
        gcc \
        g++

## Set working directory
RUN mkdir -p /software/appt
WORKDIR /software/appt

## Clone project
RUN git clone https://github.com/Bindwell/APPT /software/appt 

## Install Python requirements
RUN pip install -r requirements.txt

## Download caches.pt from Hugging Face
RUN mkdir embedding_cache_2560 && \
    cd embedding_cache_2560 && \
    wget https://huggingface.co/Bindwell/APPT/resolve/main/caches.pt?download=true \
        -O caches.pt

## Download model from Hugging Face
RUN mkdir models && \
    cd models && \
    wget https://huggingface.co/Bindwell/APPT/resolve/main/protein_protein_affinity_esm_vs_ankh_best.pt?download=true \
        -O protein_protein_affinity_esm_vs_ankh_best.pt