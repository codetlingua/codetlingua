

WORKDIR=`pwd`
export PYTHONPATH=$WORKDIR;
export PYTHONIOENCODING=utf-8;

function prompt() {
    echo;
    echo "Syntax: bash scripts/checker.sh TRANS_DIR MODEL DATASET SRC_LANG TRG_LANG TEMPERATURE N_SAMPLES";
    echo "TRANS_DIR: directory where the translations are stored";
    echo "MODEL: name of the model to use";
    echo "DATASET: name of the dataset to use";
    echo "SRC_LANG: source language";
    echo "TRG_LANG: target language";
    echo "TEMPERATURE: temperature for sampling";
    echo "N_SAMPLES: number of samples to generate";
    exit;
}

while getopts ":h" option; do
    case $option in
        h) # display help
          prompt;
    esac
done

if [[ $# < 7 ]]; then
  prompt;
fi

TRANS_DIR=$1
MODEL=$2
DATASET=$3
SRC_LANG=$4
TRG_LANG=$5
TEMPERATURE=$6
N_SAMPLES=$7

python3 tools/checker.py    --samples=$TRANS_DIR/$DATASET/$MODEL/$SRC_LANG/$TRG_LANG/temperature_$TEMPERATURE-sanitized/ \
                            --dataset=$DATASET \
                            --source_lang=$SRC_LANG \
                            --target_lang=$TRG_LANG \
                            --nsample=$N_SAMPLES
