
WORKDIR=`pwd`
export PYTHONPATH=$WORKDIR;
export PYTHONIOENCODING=utf-8;

function prompt() {
    echo;
    echo "Syntax: bash scripts/sanitize.sh MODEL DATASET SRC_LANG TRG_LANG TEMPERATURE REMOVE_PROMPT";
    echo "MODEL: name of the model to use";
    echo "DATASET: name of the dataset to use";
    echo "SRC_LANG: source language";
    echo "TRG_LANG: target language";
    echo "TEMPERATURE: temperature for sampling";
    echo "REMOVE_PROMPT: whether to remove the prompt from the generated text";
    exit;
}

while getopts ":h" option; do
    case $option in
        h) # display help
          prompt;
    esac
done

if [[ $# < 6 ]]; then
  prompt;
fi

MODEL=$1
DATASET=$2
SRC_LANG=$3
TRG_LANG=$4
TEMPERATURE=$5
REMOVE_PROMPT=$6

python3 tools/sanitize.py   --samples=translations/$DATASET/$MODEL/$SRC_LANG/$TRG_LANG/temperature_$TEMPERATURE/ \
                            --source_lang=$SRC_LANG \
                            --target_lang=$TRG_LANG \
                            --$REMOVE_PROMPT \
                            --eofs "# <END-OF-CODE>" "<END-OF-CODE>" "END-OF-CODE" "Python:" "Java:" "C:" "# Answer:" \
                            --rm-prefix-lines "Translate the above" '"""'
