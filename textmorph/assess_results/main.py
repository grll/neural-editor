from training_run import MyEditTrainingRuns
from editor import GrllNeuralEditor
from config import GrllConfig
from data_loader import GrllDataLoader
from preprocess import GrllPreprocessor
from sentence import Sentence

# Load the model to use
experiments = MyEditTrainingRuns()
exp = experiments[1]  # put here the id of the test to load

editor = GrllNeuralEditor(exp.editor)  # create custom editor with random edition.
word_vocab = exp.editor.train_decoder.word_vocab  # word vocabulary used during training of the editor.

preprocessor = GrllPreprocessor(word_vocab)  # create a preprocessor to preprocess the data.

configs = GrllConfig("textmorph/assess_results/config.json")

for config_run in configs:
    dataloader = GrllDataLoader(config_run["data_loader"]["dataset_foldername"],
                                config_run["data_loader"]["dataset_filename"],
                                config_run["data_loader"]["data_type"])

    for entities, preprocessed_sentence, data in dataloader.generate_one_preprocessed_sample(preprocessor):
        # preprocessed_sample (tuple with all necessary values)
        sentence = Sentence(preprocessed_sentence.split(" "),
                            entities=entities, original_sentence=data,
                            preprocessed_sentence=preprocessed_sentence)

        batch = [s] * config_run["edition"]["number_of_edit_vector"]

        results = Results(edit_traces)
        best_results = results.compute_best_n_results(n=5)

