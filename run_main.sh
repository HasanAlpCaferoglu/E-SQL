#!/bin/bash

# Set default values for the arguments
mode="dev"
model="gpt-4o-mini-2024-07-18"  # For GPT-4o use "gpt-4o-2024-08-06". For GPT-4o mini use "gpt-4o-mini-2024-07-18"
pipeline_order="SF-CSG-QE-SR" # First set CSG-QE-SR and then set CSG-SR

# DO NOT CHANGE THE F0LLOWING ARGUMENTS
temperature=0.0
top_p=1.0
max_tokens=4096
n=1
enrichment_level="complex"
enrichment_level_shot_number=3
enrichment_few_shot_schema_existance=False
filtering_level_shot_number=3
filtering_few_shot_schema_existance=False
cfg=True
generation_level_shot_number=3
generation_few_shot_schema_existance=False
db_sample_limit=10
relevant_description_number=20
seed=42

# Parse command line arguments to override default values
while [ "$#" -gt 0 ]; do
    case $1 in
        --mode) mode="$2"; shift ;;
        --model) model="$2"; shift ;;
        --temperature) temperature="$2"; shift ;;
        --top_p) top_p="$2"; shift ;;
        --max_tokens) max_tokens="$2"; shift ;;
        --n) n="$2"; shift ;;
        --pipeline_order) pipeline_order="$2"; shift ;;
        --enrichment_level) enrichment_level="$2"; shift ;;
        --enrichment_level_shot_number) enrichment_level_shot_number="$2"; shift ;;
        --enrichment_few_shot_schema_existance) enrichment_few_shot_schema_existance="$2"; shift ;;
        --filtering_level_shot_number) filtering_level_shot_number="$2"; shift ;;
        --filtering_few_shot_schema_existance) filtering_few_shot_schema_existance="$2"; shift ;;
        --cfg) cfg="$2"; shift ;;
        --generation_level_shot_number) generation_level_shot_number="$2"; shift ;;
        --generation_few_shot_schema_existance) generation_few_shot_schema_existance="$2"; shift ;;
        --db_sample_limit) db_sample_limit="$2"; shirt ;;
        --relevant_description_number) relevant_description_number="$2"; shirt ;;
        --seed) seed="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Run the Python script with the provided arguments
python main.py \
    --mode "$mode" \
    --model "$model" \
    --temperature "$temperature" \
    --top_p "$top_p" \
    --max_tokens "$max_tokens" \
    --n "$n" \
    --pipeline_order "$pipeline_order" \
    --enrichment_level "$enrichment_level" \
    --enrichment_level_shot_number "$enrichment_level_shot_number" \
    --enrichment_few_shot_schema_existance "$enrichment_few_shot_schema_existance" \
    --filtering_level_shot_number "$filtering_level_shot_number" \
    --filtering_few_shot_schema_existance "$filtering_few_shot_schema_existance" \
    --cfg "$cfg"\
    --generation_level_shot_number "$generation_level_shot_number" \
    --generation_few_shot_schema_existance "$generation_few_shot_schema_existance" \
    --db_sample_limit "$db_sample_limit" \
    --relevant_description_number "$relevant_description_number" \
    --seed "$seed"
