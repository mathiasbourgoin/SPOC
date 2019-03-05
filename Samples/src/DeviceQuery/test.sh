#!/bin/bash

. ../common/lib_inc.sh 

Devices=`./${1}  | grep Found | tr -d "[:alpha:]" | tr -d ":"`

([ ${Devices}  -gt "0" ] && printf "${OK}\n") || (echo "${KO}\n")
