#!/usr/bin/env bash

cp ../example/example_commandline.ipynb ./example/example_commandline.ipynb
cp ../example/example_python_api.ipynb ./example/example_python_api.ipynb
make html
mv ../docs/html ../docs_
rm -r ../docs
mv ../docs_ ../docs