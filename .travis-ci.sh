# Thanks to frenetic-lang/ocaml-topology
case "$OCAML_VERSION,$OPAM_VERSION" in
3.12.1,1.0.0) ppa=avsm/ocaml312+opam10 ;;
3.12.1,1.1.0) ppa=avsm/ocaml312+opam11 ;;
4.00.1,1.0.0) ppa=avsm/ocaml40+opam10 ;;
4.00.1,1.1.0) ppa=avsm/ocaml40+opam11 ;;
4.01.0,1.0.0) ppa=avsm/ocaml41+opam10 ;;
4.01.0,1.1.0) ppa=avsm/ocaml41+opam11 ;;
4.02.0,1.1.0) ppa=avsm/ocaml42+opam11 ;;
4.02.0,1.2.0) ppa=avsm/ocaml42+opam12 ;;
4.02.3,1.2.0) ppa=avsm/ocaml42+opam12 ;;
4.03,1.2.2) ppa=avsm/ocaml42+opam12 ;;
*) echo Unknown $OCAML_VERSION,$OPAM_VERSION; exit 1 ;;
esac

echo "yes" | sudo add-apt-repository ppa:$ppa
sudo apt-get update -qq
sudo apt-get install -qq ocaml ocaml-native-compilers camlp4-extra opam 

export OPAMYES=1
export OPAMVERBOSE=1
echo OCaml version
ocaml -version
echo OPAM versions
opam --version
opam --git-version

opam init
eval `opam config env`
case $OCAML_VERSION in
    4.02|4.02.3|4.03) opam install camlp4;;
esac

opam install ctypes ctypes-foreign
git clone https://github.com/mathiasbourgoin/OCaml_FastFlow.git && make install && cd .. 


make check
