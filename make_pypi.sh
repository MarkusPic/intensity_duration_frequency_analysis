#!/usr/bin/env bash

SCRIPTPATH=$(dirname "$BASH_SOURCE")
cd $SCRIPTPATH

python3 -m build
python3 -m twine upload dist/*