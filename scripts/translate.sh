
WORKDIR=`pwd`
export PYTHONPATH=$WORKDIR;
export PYTHONIOENCODING=utf-8;

function prompt() {
    echo;
    echo "Syntax: bash scripts/translate.sh MODEL DATASET SRC_LANG TRG_LANG PROMPT_TYPE TEMPERATURE N_SAMPLES B_SIZE MAX_LENGTH NUM_GPUS GPU_ID";
    echo "MODEL: name of the model to use";
    echo "DATASET: name of the dataset to use";
    echo "SRC_LANG: source language";
    echo "TRG_LANG: target language";
    echo "PROMPT_TYPE: prompt format";
    echo "TEMPERATURE: temperature for sampling";
    echo "N_SAMPLES: number of samples to generate";
    echo "B_SIZE: batch size";
    echo "MAX_LENGTH: maximum length of generation";
    echo "NUM_GPUS: number of GPUS to use";
    echo "GPU_ID: GPU to use";
    exit;
}

while getopts ":h" option; do
    case $option in
        h) # display help
          prompt;
    esac
done

if [[ $# < 11 ]]; then
  prompt;
fi

MODEL=$1
DATASET=$2
SRC_LANG=$3
TRG_LANG=$4
PROMPT_TYPE=$5
TEMPERATURE=$6
N_SAMPLES=$7
B_SIZE=$8
MAX_LENGTH=$9
NUM_GPUS=${10}
GPU_ID=${11}

export CUDA_VISIBLE_DEVICES=$GPU_ID;
python3 translate/translate.py --model=$MODEL --dataset=$DATASET --source_lang=$SRC_LANG --target_lang=$TRG_LANG --prompt_type=$PROMPT_TYPE --temperature=$TEMPERATURE --n_samples=$N_SAMPLES --batch_size=$B_SIZE --max_length=$MAX_LENGTH --ngpus=$NUM_GPUS --resume
