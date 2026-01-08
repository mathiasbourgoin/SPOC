FROM ocaml/opam:ubuntu-22.04-ocaml-5.4

# Install system dependencies
USER root
RUN apt-get update && apt-get install -y \
    python3-pip \
    libgmp-dev \
    pkg-config \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Jupyter
RUN pip3 install jupyterlab

# Install OCaml Jupyter Kernel and Sarek
USER opam
RUN opam update && opam install -y \
    jupyter \
    ppxlib \
    dune \
    ctypes \
    ctypes-foreign \
    alcotest

# Copy project files
COPY --chown=opam:opam . /home/opam/Sarek
WORKDIR /home/opam/Sarek

# Build and install Sarek
RUN opam exec -- dune build @install && \
    opam exec -- dune install

# Install the jupyter kernel
RUN opam exec -- ocaml-jupyter-interpreter install

# Binder expects the user to be $NB_USER (usually jovyan or opam in our case)
# And the port to be 8888
EXPOSE 8888

ENTRYPOINT ["opam", "exec", "--"]
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--NotebookApp.token=' Perkenalkan", "--NotebookApp.password=''"]
