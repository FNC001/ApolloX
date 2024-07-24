#!/bin/bash

for file in *.tar.gz; do

    directory="${file%.tar.gz}"
    mkdir -p "$directory"
    tar -xzf "$file" -C "$directory"

    if [ $? -eq 0 ]; then
        echo "Sucess: $file"
        rm "$file"
    else
        echo "Fail: $file"
    fi
done
