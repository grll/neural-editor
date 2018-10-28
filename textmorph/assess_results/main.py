from training_run import MyEditTrainingRuns
from editor import GrllNeuralEditor
from config import GrllConfig
from data_loader import GrllDataLoader
from preprocess import GrllPreprocessor

# Load the model only once to use
experiments = MyEditTrainingRuns()
exp = experiments[1]  # put here the id of the test to load

# editor = GrllNeuralEditor(exp.editor)  # create custom editor with random edition.
word_vocab = exp.editor.train_decoder.word_vocab  # word vocabulary used during training of the editor.

preprocessor = GrllPreprocessor(word_vocab)  # create a preprocessor to preprocess the data.

configs = GrllConfig()

for config_run in configs:
    dataloader = GrllDataLoader(config_run["data_loader"]["dataset_foldername"],
                                config_run["data_loader"]["dataset_filename"],
                                config_run["data_loader"]["data_type"])

    for sample in dataloader.generate_one_preprocessed_sample(preprocessor):
        # preprocess
        print(sample)
