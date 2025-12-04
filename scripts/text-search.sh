#!/bin/bash

start=$SECONDS
set -e
shopt -s expand_aliases 

while [[ "$1" == --* ]]; do
    case "$1" in
        --data_path)     
            if [[ -n "$2" && "$2" != --* ]]; then
                DATA_PATH="$2"
                shift 2
            else
                echo "Error: --data_path requires a value."
                exit 1
            fi
            ;;
        --queries)     
            if [[ -n "$2" && "$2" != --* ]]; then
                QUERIES="$2"
                shift 2
            else
                echo "Error: --queries requires a value."
                exit 1
            fi
            ;;
        --beta)     
            if [[ -n "$2" && "$2" != --* ]]; then
                BETA="$2"
                shift 2
            else
                echo "Error: --beta requires a value."
                exit 1
            fi
            ;;
        --use_lemmas)     
            LEMMAS_FLAG="--use_lemmas"
                shift
            ;;
        --use_tfidf)     
            USE_TFIDF="--use_tfidf"
                shift
            ;;
        --use_semantic)     
            USE_SEMANTIC="--use_semantic"
                shift
            ;;
        --result_file_path)     
            if [[ -n "$2" && "$2" != --* ]]; then
                RESULT_FILE_PATH="$2"
                shift 2
            else
                echo "Error: --result_file_path requires a value."
                exit 1
            fi
            ;;
        --help)
            echo "Usage: $0 --audio_dir_path <path>"
            echo "  --audio_dir_path <path>  Path to the image file (required)"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

if [[ -z "$DATA_PATH" ]]; then
    echo "Error: --data_path is required"
    echo "Use --help for usage information"
    exit 1
fi

echo "Starting text search process..."
echo "Data path: $DATA_PATH"
echo "Queries: $QUERIES"
echo "Beta: $BETA"
echo "Result file path: $RESULT_FILE_PATH"

echo "Loading modules..."
module load singularity
module load ai-tools
module load text-search

alias text-search='singularity exec --nv /cluster/tufts/apps/container/ai/images/pnadel_text-search:0.0.2.sif python'

echo "Modules loaded. Beginning text search..."
text-search /cluster/tufts/tuftsai/apps/text-search/scripts/run_text_search.py --data_path "$DATA_PATH" --queries "$QUERIES" --beta "$BETA" $LEMMAS_FLAG $USE_TFIDF $USE_SEMANTIC --result_file_path "$RESULT_FILE_PATH" || {
    echo "Error: Text search failed"
    exit 1
}

duration=$(( SECONDS - start  ))
duration_in_min=$(( duration/60  ))
echo "Text search process finished in $duration_in_min minutes!"