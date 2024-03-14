FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
# TODO: use an older system (18.04) to avoid opencv incompatibility (issue#3524)?

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
	software-properties-common python3-opencv ca-certificates python3-dev git wget sudo ninja-build tmux htop
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-distutils
RUN ln -svf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -svf /usr/bin/python3.10 /usr/bin/python

# create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
ENV SHELL=/bin/bash
RUN usermod -s /bin/bash appuser
USER appuser
WORKDIR /home/appuser

ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/pip/3.6/get-pip.py && \
	python3 get-pip.py --user && \
	rm get-pip.py
RUN /usr/bin/python3 -m pip install --upgrade pip

# install dependencies
# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN pip install --user tensorboard cmake onnx nvitop wandb==0.13.4 einops==0.5.0 pycocotools==2.0.5 scikit-image==0.19.2 joblib==1.2.0 tqdm==4.64.1 notebook==6.5.1 pandas==1.5.1 seaborn==0.12.1 numpy>=1.22 ipdb==0.13.9  # cmake from apt-get is too old
RUN pip install --user torch==1.12.0+cu113 torchvision==0.13.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113


# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"
# This will by default build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
