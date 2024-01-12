

WORKDIR=`pwd`
export PYTHONPATH=$WORKDIR;
export PYTHONIOENCODING=utf-8;

function prompt() {
    echo;
    echo "Syntax: bash scripts/checker.sh MODEL DATASET SRC_LANG TRG_LANG TEMPERATURE";
    echo "MODEL: name of the model to use";
    echo "DATASET: name of the dataset to use";
    echo "SRC_LANG: source language";
    echo "TRG_LANG: target language";
    echo "TEMPERATURE: temperature for sampling";
    exit;
}

while getopts ":h" option; do
    case $option in
        h) # display help
          prompt;
    esac
done

if [[ $# < 5 ]]; then
  prompt;
fi

MODEL=$1
DATASET=$2
SRC_LANG=$3
TRG_LANG=$4
TEMPERATURE=$5

python3 tools/checker.py    --samples=translations/$DATASET/$MODEL/$SRC_LANG/$TRG_LANG/temperature_$TEMPERATURE-sanitized/ \
                            --dataset=$DATASET \
                            --source_lang=$SRC_LANG \
                            --target_lang=$TRG_LANG \
                            --nsample=20
