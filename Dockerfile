FROM continuumio/miniconda3:4.5.12

# Install build essentials and clean up
RUN apt-get update --quiet \
  && apt-get install -y --no-install-recommends --quiet build-essential \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Update conda, install packages, and clean up
RUN conda install python=3.6 --yes --quiet \
  && conda clean --yes --all \
  && hash -r

# Install the main library
COPY . /code

RUN cd /code \
  && pip install --upgrade pip \
  && pip install numpy \
  && pip install -e .[test,ssm_gpy,ssm_pytorch] \
  && rm -rf /root/.cache

WORKDIR /code
