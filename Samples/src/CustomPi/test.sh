#!/bin/bash

. ../common/lib_inc.sh 

printf "\n  + GPU_COMPUTATION:"

res1=`./${1} grep "GPU Computation" ${res} | tr -d "[:alpha:]" | tr -d ":"`

if `echo "${res1}" | grep -q 3\.` ; then
    printf "${OK}\n"
else
    printf "${KO}\n"
fi

printf "  + GPU_COMPUTATION WITH COMPLEX32:"

res2=`./${1} grep "GPU Complex" ${res} | tr -d "[:alpha:]" | tr -d ":" `


if `echo "${res2}" | grep -q 3\.` ; then
    printf "${OK}\n"
else
    printf "${KO}\n"
fi
