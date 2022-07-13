#!/bin/bash

pip install -r requirements.txt

# install spacy models
python -m spacy download en_core_web_sm
# Or manually download from https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.3.0/en_core_web_sm-3.3.0.tar.gz
#pip install cache/en_core_web_sm-3.3.0.tar.gz
