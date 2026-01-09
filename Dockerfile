FROM ocaml/opam:ubuntu-22.04-ocaml-5.4

USER root
RUN apt-get update && apt-get install -y \
    python3-pip libgmp-dev pkg-config libffi-dev m4 libzmq3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install jupyterlab thebe

ARG NB_USER=opam
ARG NB_UID=1000
ENV USER ${NB_USER}
ENV HOME /home/${NB_USER}

USER opam
WORKDIR ${HOME}/Sarek

# 1. Update opam and install build deps
RUN opam update
RUN opam exec -- opam install -y dune ppxlib ctypes ctypes-foreign alcotest ocamlfind ocaml-compiler-libs

# 2. Install Jupyter and ZeroMQ
RUN opam exec -- opam install -y conf-zmq zmq zmq-lwt jupyter

# 3. Copy and build the project
COPY --chown=opam:opam . .
RUN opam exec -- dune build @install && \
    opam exec -- dune install

# 4. Final configuration
RUN opam exec -- ocaml-jupyter-opam-genspec && \
    opam exec -- jupyter kernelspec install --user --name ocaml-jupyter $(opam exec -- opam var share)/jupyter

# 5. Fix for OCaml 5 Effects in Toplevel
# We ensure topfind is loaded and we add the stdlib directory to search path
RUN echo '#use "topfind";;' > ${HOME}/.ocamlinit && \
    echo '#directory "^";;' >> ${HOME}/.ocamlinit

EXPOSE 8888
ENTRYPOINT ["opam", "exec", "--"]
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''"]