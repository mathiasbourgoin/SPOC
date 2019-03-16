#!/bin/bash

. ../common/lib_inc.sh 

printf "\n  + GPU_COMPUTATION:"

res1=`./${1} | grep "GPU Computation"` 


if ` grep -q "= 3\." <<< ${res1}` ; then
    printf "${OK}\n"
else
    printf "${KO}\n"
fi

printf "  + GPU_COMPUTATION WITH COMPLEX32:"

res2=`./${1} | grep "GPU Complex"` 


if `grep -q "= 3\." <<< ${res2}` ; then
    printf "${OK}\n"
else
    printf "${KO}\n"
fi
