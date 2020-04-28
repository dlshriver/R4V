#!/bin/bash
set -e

python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools

while read req || [ -n "$req" ]
do
    echo "pip install $req"
    pip install $req
done < requirements.txt
