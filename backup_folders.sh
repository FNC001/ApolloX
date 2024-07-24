#!/bin/bash
for dir in */; do
  archive_name="${dir%/}.tar.gz"
  tar -czvf "$archive_name" "$dir"
done
