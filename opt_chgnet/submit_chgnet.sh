#!/bin/bash
find . -type d \( ! -name . \) -print | while read dir
do
    cd "$dir"
    nohup python -u run_chg_gpu.py > out.log 2>&1 &
    cd -
done
