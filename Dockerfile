FROM ocaml/opam:ubuntu-22.04-ocaml-5.4

USER root
# Install critical system deps
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

# 1. Update opam
RUN opam update

# 2. Install base build tools
RUN opam exec -- opam install -y dune ppxlib ctypes ctypes-foreign alcotest ocamlfind

# 3. Install Jupyter and ZeroMQ bindings
RUN opam exec -- opam install -y conf-zmq zmq zmq-lwt jupyter

# 4. Copy and build the project
COPY --chown=opam:opam . .

RUN opam exec -- dune build @install && \
    opam exec -- dune install

# 5. Final configuration - register the kernel
RUN opam exec -- ocaml-jupyter-opam-genspec && \
    opam exec -- jupyter kernelspec install --user --name ocaml-jupyter $(opam exec -- opam var share)/jupyter

# 6. Ensure Topfind is loaded for #require support
RUN echo '#use "topfind";;' >> ${HOME}/.ocamlinit

EXPOSE 8888
ENTRYPOINT ["opam", "exec", "--"]
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''"]
