#!/bin/sh

# first time need to create the venv
# python3 -m venv venv_05

source venv_05/bin/activate
cd localGPT_orig_reqtxt/

# first time just really ensure it gets a force-reinstall and upgrade
# python -m pip install -r requirements.txt --force-reinstall --upgrade

python -m pip install -r requirements.txt

# get nvidia support
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade