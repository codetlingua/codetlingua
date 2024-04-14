
WORKDIR=`pwd`
export PYTHONPATH=$WORKDIR;
export PYTHONIOENCODING=utf-8;

function prompt() {
    echo;
    echo "Syntax: bash scripts/sanitize.sh TRANS_DIR MODEL DATASET SRC_LANG TRG_LANG TEMPERATURE REMOVE_PROMPT";
    echo "TRANS_DIR: directory where the translations are stored";
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

if [[ $# < 7 ]]; then
  prompt;
fi

TRANS_DIR=$1
MODEL=$2
DATASET=$3
SRC_LANG=$4
TRG_LANG=$5
TEMPERATURE=$6
REMOVE_PROMPT=$7

python3 tools/sanitize.py   --samples=$TRANS_DIR/$DATASET/$MODEL/$SRC_LANG/$TRG_LANG/temperature_$TEMPERATURE/ \
                            --source_lang=$SRC_LANG \
                            --target_lang=$TRG_LANG \
                            --$REMOVE_PROMPT \
                            --eofs "# <END-OF-CODE>" "// <END-OF-CODE>" "<END-OF-CODE>" "END-OF-CODE" "Python:" "Java:" "C:" "C++:" "Go:" "# Answer:" \
                                   "# 2nd solution:" "Note" "Input:" "1." "Explanation" "A:"\
                            --rm-prefix-lines "Translate the above" '"""' "Here" "[PYTHON]" "[JAVA]" "[C]" "[C++]" "[GO]" "C++" "cpp" "c++" "C" "Python" "python" "Java" "java" "Go" "go" "The" "Please" "This" "That" "After" "You" "-" "In" "Also" "To" "Key"
