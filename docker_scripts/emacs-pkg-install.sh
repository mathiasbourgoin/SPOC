#!/bin/sh

if [ $# -ne 1 ]
then
  echo "Usage: `basename $0` <package>"
  exit 1
fi

emacs --batch --eval "(defconst pkg-to-install '$1)" -l emacs-pkg-install.el
