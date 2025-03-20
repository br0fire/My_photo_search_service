#!/bin/bash

unzip Dataset_part1.zip -d dataset/
unzip Dataset_part2.zip -d dataset/
unzip Dataset_part3.zip -d dataset/

rm -rf dataset/__MACOSX

src="dataset"
dst="dataset1"

mkdir -p "$dst"

find "$src" -type f | while read -r file; do
    base=$(basename "$file")
    target="$dst/$base"

    # If file already exists, append a numeric suffix
    if [[ -e "$target" ]]; then
        i=1
        ext="${base##*.}"
        name="${base%.*}"

        # Handle files without an extension
        if [[ "$name" == "$ext" ]]; then
            ext=""
        else
            ext=".$ext"
        fi

        while [[ -e "$dst/${name}_$i$ext" ]]; do
            ((i++))
        done

        target="$dst/${name}_$i$ext"
    fi

    mv "$file" "$target"
done
