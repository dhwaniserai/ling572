#!/bin/sh
if (($#<6))
then
    /nopt/python-3.6/bin/python3 grad_desc.py.py $1 $2 $3 $4
else
    /nopt/python-3.6/bin/python3 grad_desc.py.py $1 $2 $3 $4 $5
fi