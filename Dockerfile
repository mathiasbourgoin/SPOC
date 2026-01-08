FROM ocaml/opam:ubuntu-22.04-ocaml-5.4

USER root
RUN apt-get update && apt-get install -y \
    python3-pip libgmp-dev pkg-config libffi-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install jupyterlab thebe

USER opam
# 1. Cache dependencies (heavy part)
RUN opam update && opam install -y \
    jupyter \
    ppxlib \
    dune \
    ctypes \
    ctypes-foreign \
    alcotest \
    conf-pkg-config \
    conf-libffi

# 2. Copy and build the project (light part)
COPY --chown=opam:opam . /home/opam/Sarek
WORKDIR /home/opam/Sarek

RUN opam exec -- dune build @install && \
    opam exec -- dune install

# 3. Final configuration
RUN opam exec -- ocaml-jupyter-interpreter install

EXPOSE 8888
ENTRYPOINT ["opam", "exec", "--"]
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--NotebookApp.token='\'", "--NotebookApp.password=''"]