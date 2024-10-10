#!/bin/bash

json_file_path=$1
target_dir=$2


mkdir -p "$target_dir"

process_line() {
    line="$1"
    #echo "Processing line: $line"  # Debugging line

    # Extract text and filename using awk
    text=$(echo "$line" | awk -F 'text: ' '{print $2}' | awk -F 'duration:' '{print $1}' | tr -d '",{}')
    file_path=$(echo "$line" | awk -F 'audio_filepath: ' '{print $2}' |awk -F 'duration: ' '{print $1}' | tr -d '",{}')
    if [[ -z "$file_path" || -z "$text" ]]; then
        echo "Error parsing line: $line"
        return 1
    fi

    file_name=$(basename "$file_path")
    file_name="${file_name%.*}.txt"  # Replace extension with .txt
    # Check for write permissions in target directory
    if [[ ! -w "$target_dir" ]]; then
        echo "No write permission in $target_dir"
        return 1
    fi

    echo "$text" > "$target_dir/$file_name"
}

export -f process_line

# Get number of CPU cores
num_cores=$(python -c "import multiprocessing; print(multiprocessing.cpu_count())")

# Use xargs for parallel processing
cat "$json_file_path" | xargs -I {} -P $num_cores bash -c 'process_line "$@"' _ {}
