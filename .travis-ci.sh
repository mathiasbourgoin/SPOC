
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_10.0.130-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1404_10.0.130-1_amd64.deb
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/7fa2af80.pub
sudo apt-get update -qq
sudo apt-get install -qq build-essential libffi-dev pkg-config cuda-nvrtc-dev-10-0
sudo mkdir -p /usr/local/cuda-10.0/include/
sudo ln -s /usr/local/cuda-10.0/targets/x86_64-linux/include/nvrtc.h /usr/local/cuda-10.0/include/nvrtc.h

export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64\
       ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export C_INCLUDE_PATH=/usr/local/cuda-10.0/targets/x86_64-linux/include/${C_INCLUDE_PATH:+:${C_INCLUDE_PATH}}
export CUDA_PATH=/usr/local/cuda-10.0



sudo .travis/install_intel_opencl.sh

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


make check

cd Samples/src/DeviceQuery && make && ./DeviceQuery.asm


