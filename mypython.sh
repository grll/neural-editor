#!/bin/bash -l
$(cat /etc/environement | xargs)
/opt/conda/envs/pytorch-py27/bin/python "$@"