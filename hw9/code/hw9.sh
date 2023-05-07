#!/bin/sh
# Usage: hw9.sh CONFIG_FILE OUTPUT_FILE

source ~/anaconda3/bin/activate "/dropbox/22-23/572/env"

time python hw9_script.py $1 > $2 
