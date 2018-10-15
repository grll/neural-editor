import sys,os
sys.path.append(os.getcwd())

import argparse

from gtd.log import set_log_level
from textmorph.text_generation.training_run import EditTrainingRuns
from textmorph import data
from textmorph.text_generation.training_run import GenData
from gtd.ml.torch.utils import similar_size_batches

set_log_level('DEBUG')

# Reload experiement 6
experiments = EditTrainingRuns(check_commit=False)
exp_id = [6]
exp = experiments[int(exp_id[0])]

# Load the model trained
print(exp.editor)

# Load sources sentences
data_dir = os.path.join(data.workspace.root, "textgen_dataset")
source_data = GenData(data_dir) # load the data

batches = similar_size_batches(source_data.data, 32, size=lambda x: len(x))

for batch in verboserate(batches, desc='Streaming Source Sentences'):
    # Source Encode
    # Edit Encode
    # Decode with attention using generated tokens from previous timestep as input in first lstm layer.
    # Modify the model to generate sentences as intended (no target sentences)
