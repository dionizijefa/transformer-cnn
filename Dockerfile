FROM tensorflow/tensorflow:1.12.0-rc1-gpu-py3

RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.4 \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

ENV PATH /opt/conda/bin:$PATH

RUN conda create -n myenv && \
    echo "source activate myenv" > ~/.bashrc

SHELL ["/bin/bash", "--login", "-c"]

RUN source activate myenv && \
    conda config --add channels rdkit && \
    conda install -y rdkit=2018.09.2

RUN source activate myenv && \
    pip install --no-cache-dir \
    molvs \
    numpy

WORKDIR /app

COPY . .

ENTRYPOINT ["conda", "run", "-n", "myenv"]
CMD ["python3", "transformer-cnn.py", "config.cfg"] 