#!/bin/bash

for file in *.tar.gz; do
    tar -xzf "$file"

    if [ $? -eq 0 ]; then
        echo "解压成功: $file"
        rm "$file"
    else
        echo "解压失败: $file"
    fi
done
