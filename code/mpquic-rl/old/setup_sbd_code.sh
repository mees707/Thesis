#!/bin/bash

Run="git clone https://github.com/thomaswpp/mpquic-sbd.git &&\
cd mpquic-sbd &&\
./build.sh \
"
bash ./ssh_exec.sh "$Run"