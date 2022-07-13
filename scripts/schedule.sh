#!/bin/bash
# Shedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

python train.py experiment=bert4gcn datamodule.dataset=twitter model.window=2
python train.py experiment=bert4gcn datamodule.dataset=laptop model.window=3
python train.py experiment=bert4gcn datamodule.dataset=restaurant model.window=3


