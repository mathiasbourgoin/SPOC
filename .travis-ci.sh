
.travis/install_cuda.sh

export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64\
       ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export C_INCLUDE_PATH=/usr/local/cuda-10.0/targets/x86_64-linux/include/${C_INCLUDE_PATH:+:${C_INCLUDE_PATH}}
export CUDA_PATH=/usr/local/cuda-10.0


.travis/install_intel_opencl.sh

.travis/install_ocaml.sh

eval `${HOME}/opam config env`
make check


