
function setup_env() {
    conda create -n plempirical python=3.10.13;       # miniconda 23.5.2 download from https://docs.conda.io/en/latest/miniconda_hashes.html
    conda activate plempirical;
}

function start_docker() {
    docker build -t codecombat:v0 .
    docker run -it codecombat:v0
}

function install_dependencies() {
    pip3 install openai==1.6.1;
    pip3 install torch==2.1.2;
    pip3 install vllm==0.2.7;
    pip3 install transformers==4.36.2;
    pip3 install tokenizers==0.15.0;
    pip3 install rich==13.3.5;
    pip3 install datasets==2.16.1;
    pip3 install termcolor==2.4.0;
}

# setup_env;
# start_docker;
# install_dependencies;
