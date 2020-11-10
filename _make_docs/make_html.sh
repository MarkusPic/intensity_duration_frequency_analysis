#!/usr/bin/env bash

#rm -r ./example
#cp ../example ./example -r
#cp ../README.md ./README.md
make html
mv ../docs/html ../docs_
rm -r ../docs
mv ../docs_ ../docs