FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    wget \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

RUN conda create -n myenv python=3.6 && \
    echo "source activate myenv" > ~/.bashrc

SHELL ["/bin/bash", "--login", "-c"]

RUN source activate myenv && \
    conda config --add channels rdkit && \
    conda install -y tensorflow && \
    conda install -y rdkit && \
    conda install -y numpy h5py

WORKDIR /app

COPY . .

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/compat

CMD ["/bin/bash", "-l"]