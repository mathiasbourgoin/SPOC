FROM ocaml/opam:ubuntu-22.04-ocaml-5.4

USER root
# Install critical system deps
RUN apt-get update && apt-get install -y \
    python3-pip libgmp-dev pkg-config libffi-dev m4 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install jupyterlab thebe

USER opam
WORKDIR /home/opam/Sarek

# 1. Update opam
RUN opam update

# 2. Install base build tools (fast)
RUN opam exec -- opam install -y dune ppxlib ctypes ctypes-foreign alcotest

# 3. Install Jupyter kernel (this is the most likely to fail/conflict)
# We use --best-effort to help the solver if needed
RUN opam exec -- opam install -y jupyter

# 4. Copy and build the project
COPY --chown=opam:opam . .

RUN opam exec -- dune build @install && \
    opam exec -- dune install

# 5. Final configuration
RUN opam exec -- ocaml-jupyter-interpreter install

# Binder compatibility: use UID 1000 (which 'opam' user already is)
# We don't change the username to jovyan to avoid breaking opam paths, 
# but UID 1000 is what matters for Binder.

EXPOSE 8888
ENTRYPOINT ["opam", "exec", "--"]
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''"]
