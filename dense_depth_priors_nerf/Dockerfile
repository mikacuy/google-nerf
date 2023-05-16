FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
ENV LANG=C.UTF-8

RUN mkdir /src/
WORKDIR /src

# Create conda environment
COPY env.yml /workspace/env.yaml
RUN conda update -n base conda
RUN conda env create -f /workspace/env.yaml
RUN chmod 777 -R /opt/conda/envs/scade/

# Copy code into source directory
COPY data /src/data
COPY depth_priors_network_weights /src/depth_priors_network_weights
COPY metric /src/metric
COPY model /src/model
COPY train_utils /src/train_utils
COPY *.py /src/
RUN chmod 777 -R /src/

# Create entry points
# ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "scade"]

# Create entry points
RUN conda activate scade
