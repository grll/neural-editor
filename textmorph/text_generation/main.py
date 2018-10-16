import sys,os
sys.path.append(os.getcwd())

import argparse

from gtd.log import set_log_level
from textmorph.text_generation.training_run import EditTrainingRuns
from textmorph import data
from textmorph.text_generation.training_run import GenData
from gtd.ml.torch.utils import similar_size_batches
from gtd.chrono import verboserate

set_log_level('DEBUG')

# Reload experiment 6
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
    # Source & Edit Encoding
    source_words, insert_words, insert_exact_words, delete_words, delete_exact_words, target_words, edit_embed = exp.editor._batch_editor_examples(batch)
    encoder_input = exp.editor.encoder.preprocess(source_words, insert_words, insert_exact_words, delete_words, delete_exact_words, edit_embed)
    encoder_output = exp.editor.encoder(encoder_input)

    # Decoding
    
    break
    # Edit Encode
    # Decode with attention using generated tokens from previous timestep as input in first lstm layer.
