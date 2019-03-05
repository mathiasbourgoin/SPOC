#!/bin/bash

. ../common/lib_inc.sh 

if (yes Y | ./${1}) | grep -q OK; then
    printf "${OK}\n"
else
    printf "${KO}\n"
fi


