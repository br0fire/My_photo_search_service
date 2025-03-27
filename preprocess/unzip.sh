#!/bin/bash

set -e  # Exit immediately on error

# Usage message
usage() {
    echo "Usage: $0 --zip1 file1.zip --zip2 file2.zip --zip3 file3.zip --src source_folder --dst destination_folder"
    exit 1
}

# Default values
src=""
dst=""
zip1=""
zip2=""
zip3=""

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --zip1) zip1="$2"; shift ;;
        --zip2) zip2="$2"; shift ;;
        --zip3) zip3="$2"; shift ;;
        --src) src="$2"; shift ;;
        --dst) dst="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# Validate required arguments
if [[ -z "$zip1" || -z "$zip2" || -z "$zip3" || -z "$src" || -z "$dst" ]]; then
    echo "Error: Missing one or more required arguments."
    usage
fi

# Unzip the datasets
unzip "$zip1" -d "$src/"
unzip "$zip2" -d "$src/"
unzip "$zip3" -d "$src/"

# Remove macOS metadata folder if it exists
rm -rf "$src/__MACOSX"

# Create destination folder
mkdir -p "$dst"

# Move files and rename if duplicates exist
find "$src" -type f | while read -r file; do
    base=$(basename "$file")
    target="$dst/$base"

    if [[ -e "$target" ]]; then
        i=1
        ext="${base##*.}"
        name="${base%.*}"

        # Handle files without extensions
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
