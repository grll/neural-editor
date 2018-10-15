import sys,os
sys.path.append(os.getcwd())

import argparse

from gtd.io import save_stdout
from gtd.log import set_log_level
from gtd.utils import Config
from textmorph.text_generation.training_run import EditTrainingRuns

set_log_level('DEBUG')

# Reload experiement 6
experiments = EditTrainingRuns(check_commit=False)
exp_id = [6]
exp = experiments[int(exp_id[0])]

# Load the model trained
print(exp.editor)
# Load the source sentence
# Modify the model to generate sentences as intended (no target sentences)
