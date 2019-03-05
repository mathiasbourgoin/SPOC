#!/bin/bash

set -ev

###########################
# Install OCaml environment
###########################

export OPAMYES=1
export OPAMJOBS=2
export OPAMVERBOSE=1

wget -O ${HOME}/opam https://github.com/ocaml/opam/releases/download/2.0.2/opam-2.0.2-x86_64-linux

chmod +x ${HOME}/opam
${HOME}/opam init --compiler=${OCAML_VERSION} --disable-sandboxing
eval `${HOME}/opam config env`
${HOME}/opam update --upgrade

echo OCaml version
ocaml -version
echo OPAM versions
${HOME}/opam --version
${HOME}/opam --git-version

${HOME}/opam install camlp4 ctypes ctypes-foreign cppo
