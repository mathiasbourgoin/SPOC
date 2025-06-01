FROM ubuntu:20.04
MAINTAINER Mathias Bourgoin <mathias.bourgoin@gmail.com>
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -y update && \
    apt-get -y install sudo pkg-config git build-essential \
    m4 software-properties-common aspcud unzip curl \
    libx11-dev \
    libffi-dev emacs pkg-config wget opam gnupg \
    ocl-icd-opencl-dev intel-opencl-icd libclc-dev \
    wget gnupg
# OCaml will be installed via opam

# Install CUDA Toolkit
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get -y install cuda-toolkit-11-8 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm cuda-keyring_1.1-1_all.deb

RUN useradd -ms /bin/bash spoc && echo "spoc:spoc" | chpasswd && adduser spoc sudo
USER spoc
WORKDIR /home/spoc
CMD /bin/bash

# opam configuration
RUN opam init -a --disable-sandboxing --root /home/spoc/.opam --yes && \
    opam switch create 4.12.0 --yes && \
    eval $(opam env) && opam update --yes && \
    opam depext conf-pkg-config.1.0 --yes && \
    opam install camlp4 ctypes ocp-indent ctypes-foreign ocamlfind cppo merlin tuareg --yes


RUN rm -rf SPOC
RUN git clone https://github.com/mathiasbourgoin/SPOC.git

ADD docker_scripts/.bashrc /home/spoc/.bashrc

ENV CUDA_PATH=/usr/local/cuda-11.8
ENV PATH=$CUDA_PATH/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

WORKDIR SPOC/Spoc
RUN eval $(opam env) && make && \
    ocamlfind install spoc *.cma *.a *.so *.cmxa *.cmi META  && \
    cd extension && make && make install

WORKDIR ../SpocLibs/Sarek
RUN eval $(opam env) && make && make install

RUN mkdir /home/spoc/emacs_install
ADD docker_scripts/emacs-pkg-install.el  /home/spoc/emacs_install/emacs-pkg-install.el
ADD docker_scripts/emacs-pkg-install.sh  /home/spoc/emacs_install/emacs-pkg-install.sh

WORKDIR /home/spoc/emacs_install

RUN (sh ./emacs-pkg-install.sh auto-complete && \
    sh ./emacs-pkg-install.sh company && \
    sh ./emacs-pkg-install.sh company-irony) || echo "No emacs package installed"

ADD docker_scripts/.emacs /home/spoc/.emacs

WORKDIR /home/spoc
