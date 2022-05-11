FROM nvcr.io/nvidia/pytorch:22.03-py3 as base

RUN apt-get update && apt-get install --no-install-recommends -y \
  git \
  curl \
  unzip \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /work

FROM base as jupyter

# for jupyterlab_vim
RUN curl -sL https://deb.nodesource.com/setup_16.x | bash -
RUN apt-get update \
  && apt-get install -y nodejs \
  && rm -rf /var/lib/apt/lists/*

RUN pip install jupyterlab==3.3.2

RUN jupyter labextension install @axlair/jupyterlab_vim
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager

COPY jupyter-config/jupyter_lab_config.py /root/.jupyter/
COPY jupyter-config/overrides.json  /root/.jupyter/

RUN APPLICATION_DIR=$(jupyter lab path | grep "Application" | cut -d ":" -f 2 | tr -d " ") && \
  mv /root/.jupyter/overrides.json ${APPLICATION_DIR}/settings/overrides.json

# 大きいpackageは別でinstallしておく
RUN pip install transformers \
  opencv-python \
  opencv-contrib-python

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

USER root
ENV PYTHONPATH="/work:$PYTHONPATH"

CMD ["jupyter", "lab"]
