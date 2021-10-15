FROM nvcr.io/nvidia/pytorch:20.12-py3

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \ 
	wget \ 
  curl \
	git \
	neovim \
	zip
RUN conda install -c conda-forge \
	pillow \
	numpy \
	tensorboard \
	scipy \
	gym-atari \
	yapf \
	filelock \
	pip

RUN git clone https://github.com/neevparikh/pfrl && cd pfrl && \
  pip install -e .
RUN git clone https://github.com/rgreenblatt/learning_from_human_preferences

RUN touch /root/.no_auto_tmux
WORKDIR /root/learning_from_human_preferences
