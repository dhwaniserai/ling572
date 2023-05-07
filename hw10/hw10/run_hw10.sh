#!/bin/sh
source ~/anaconda3/bin/activate "/dropbox/22-23/572/hw10/code/env"
# if you install miniconda in a different directory, try the following command
# source path_to_miniconda/miniconda3/bin/activate "/dropbox/22-23/572/env"

cd .

#python main.py --num_epochs 6 --data_dir /dropbox/22-23/572/hw10/code/data/

# q3
python main.py --num_epochs 6 --data_dir /dropbox/22-23/572/hw10/code/data/ > q3.out

# q4
python main.py --num_epochs 6 --data_dir /dropbox/22-23/572/hw10/code/data/ --L2 > q4.out

# q5
python main.py --num_epochs 12 --data_dir /dropbox/22-23/572/hw10/code/data/ --patience 3 --L2 > q5.out


