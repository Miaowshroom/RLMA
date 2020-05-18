FROM continuumio/miniconda3
COPY env.yml ./tmp/env.yml
RUN conda env create -f ./tmp/env.yml
RUN echo "source activate RLMA" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH
WORKDIR /data
