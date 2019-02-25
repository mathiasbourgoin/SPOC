FROM ubuntu:16.10
MAINTAINER Mathias Bourgoin <mathias.bourgoin@gmail.com>
RUN apt-get -y update && \
#RUN apt-get -y install sudo pkg-config git build-essential m4 software-properties-common aspcud unzip curl libx11-dev ocaml ocaml-native-compilers camlp4-extra git libffi-dev emacs pkg-config wget aspcud
     apt-get -y install sudo pkg-config git build-essential \
     m4 software-properties-common aspcud unzip curl \
     libx11-dev ocaml ocaml-native-compilers camlp4-extra \
     git libffi-dev emacs pkg-config wget aspcud
#RUN apt-get install -y git libffi-dev emacs pkg-config wget aspcud
#RUN apt-get install -y emacs
#RUN apt-get install -y pkg-config
#RUN apt-get install -y wget aspcud

RUN git clone https://github.com/mathiasbourgoin/amd_sdk.git

RUN sh amd_sdk/amd_sdk.sh

RUN apt-get install -y opam && \
    useradd -ms /bin/bash spoc && echo "spoc:spoc" | chpasswd && adduser spoc sudo
#RUN useradd -ms /bin/bash spoc && echo "spoc:spoc" | chpasswd && adduser spoc sudo
USER spoc
WORKDIR /home/spoc
CMD /bin/bash


#RUN wget https://raw.github.com/ocaml/opam/master/shell/opam_installer.sh -O - | sh -s ~/.opam

RUN opam init -a --root /home/spoc/.opam && \
    opam switch 4.03.0	&& \
#RUN opam switch 4.03.0
#RUN eval `opam config env`
#RUN eval `opam config env`&& opam update && \
    eval `opam config env`&& opam update && \
    opam depext conf-pkg-config.1.0 && \
    opam install camlp4 ctypes ocp-indent ctypes-foreign ocamlfind cppo
#RUN eval `opam config env` && opam depext conf-pkg-config.1.0
#RUN eval `opam config env` && opam install camlp4 ctypes ocp-indent ctypes-foreign ocamlfind
#RUN eval `opam config env` && opam install ctypes
#RUN eval `opam config env` && opam install ocp-indent
#RUN eval `opam config env` && opam install ctypes-foreign
#RUN opam install merlin
#RUN eval `opam config env` && opam install ocamlfind


RUN rm -rf SPOC
RUN git clone https://github.com/mathiasbourgoin/SPOC.git

ADD docker_scripts/.bashrc /home/spoc/.bashrc

WORKDIR SPOC/Spoc
RUN eval `opam config env` && make && \
    ocamlfind install spoc *.cma *.a *.so *.cmxa *.cmi META  && \
    cd extension && make && make install
#RUN eval `opam config env` && ocamlfind install spoc *.cma *.a *.so *.cmxa *.cmi META
#RUN cd extension && eval `opam config env` && make && make install
#RUN cd extension && eval `opam config env` && make install 

WORKDIR ../SpocLibs/Sarek
RUN eval `opam config env` && make && make install 


RUN mkdir /home/spoc/emacs_install
ADD docker_scripts/emacs-pkg-install.el  /home/spoc/emacs_install/emacs-pkg-install.el
ADD docker_scripts/emacs-pkg-install.sh  /home/spoc/emacs_install/emacs-pkg-install.sh

WORKDIR /home/spoc/emacs_install

RUN (sh ./emacs-pkg-install.sh auto-complete && \
    sh ./emacs-pkg-install.sh company && \
    sh ./emacs-pkg-install.sh company-irony) || echo "No emacs package installed"
#RUN sh ./emacs-pkg-install.sh company
#RUN sh ./emacs-pkg-install.sh company-irony
#RUN eval `opam config env`&& opam install merlin tuareg ocp-indent
#RUN eval `opam config env`&& opam install tuareg
#RUN eval `opam config env`&& opam install ocp-indent

ADD docker_scripts/.emacs /home/spoc/.emacs

WORKDIR /home/spoc
