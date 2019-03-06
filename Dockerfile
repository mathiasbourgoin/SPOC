FROM ubuntu:18.10
MAINTAINER Mathias Bourgoin <mathias.bourgoin@gmail.com>
RUN  apt-get -yq update

RUN  apt-get -yq install --no-install-recommends        \
     --allow-downgrades --allow-remove-essential        \
     --allow-change-held-packages                       \
     sudo pkg-config git build-essential                \
     software-properties-common unzip curl              \
     libx11-dev tar apt-utils m4 dirmngr gpg-agent      \
     libffi-dev emacs-nox wget && apt-get -yq update

RUN useradd -ms /bin/bash spoc && echo "spoc:spoc" | chpasswd && adduser spoc sudo

WORKDIR /home/spoc

RUN rm -rf SPOC && git clone https://github.com/mathiasbourgoin/SPOC.git

WORKDIR SPOC

RUN .travis/install_intel_opencl.sh
RUN .travis/install_cuda.sh

USER spoc
CMD /bin/bash

RUN cp docker_scripts/.bashrc /home/spoc/.bashrc


RUN .travis/install_ocaml.sh

USER root
RUN chown -R spoc /home/spoc/SPOC
USER spoc

RUN eval `/home/spoc/opam config env` && make install install_sarek 

RUN mkdir -p /home/spoc/emacs_install
RUN cp /home/spoc/SPOC/docker_scripts/emacs-pkg-install.el  /home/spoc/emacs_install/emacs-pkg-install.el
RUN cp /home/spoc/SPOC/docker_scripts/emacs-pkg-install.sh  /home/spoc/emacs_install/emacs-pkg-install.sh

WORKDIR /home/spoc/emacs_install

RUN (sh ./emacs-pkg-install.sh auto-complete && \
    sh ./emacs-pkg-install.sh company && \
    sh ./emacs-pkg-install.sh company-irony) || echo "No emacs package installed"

RUN cp  docker_scripts/.emacs /home/spoc/.emacs

WORKDIR /home/spoc
