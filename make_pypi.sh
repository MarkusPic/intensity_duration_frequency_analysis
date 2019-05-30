#!/usr/bin/env bash

python3 setup.py bdist_wheel
python3 -m twine upload dist/*