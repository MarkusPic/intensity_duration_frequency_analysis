#!/usr/bin/env bash

#python3 setup.py bdist_wheel
#python3 -m twine upload dist/*

# ____________________

SCRIPTPATH=$(dirname "$BASH_SOURCE")
cd $SCRIPTPATH

python3 -m build
python3 -m twine upload dist/*