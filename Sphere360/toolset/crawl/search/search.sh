#!/bin/bash
num_pages=1 # Number of search pages (max 50 results per page)

keyword_file='' # Keyword file  
postfix=" spatial audio 360" # Keyword suffix, e.g., "spatial audio 360"
tmp_dir="tmp/" # Temporary folder for search results  
output_dir="search_result/" # Final output folder for all search results  

log_file="search.log" # Log file  

# Search and output initial results  
python search.py \
    -i "$keyword_file" \
    -o "$output_dir" \
    -n "$num_pages" \
    -p "$postfix" \
    -t "$tmp_dir" \
    --aid-id "$aid_id" \
    > $log_file
