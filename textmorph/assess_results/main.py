from training_run import MyEditTrainingRuns
from editor import GrllNeuralEditor
from config import GrllConfig
from data_loader import GrllDataLoader
from preprocess import GrllPreprocessor
from sentence import Sentence
from results import Results
from textmorph import data
from os.path import dirname, join

# Load the model to use
experiments = MyEditTrainingRuns()
exp = experiments[6]  # put here the id of the test to load

editor = GrllNeuralEditor(exp.editor)  # create custom editor with random edition.
word_vocab = exp.editor.train_decoder.word_vocab  # word vocabulary used during training of the editor.

preprocessor = GrllPreprocessor(word_vocab)  # create a preprocessor to preprocess the data.

configs = GrllConfig("textmorph/assess_results/config.json")

for config_run in configs:
    dataloader = GrllDataLoader(config_run["data_loader"]["dataset_foldername"],
                                config_run["data_loader"]["dataset_filename"],
                                config_run["data_loader"]["data_type"])

    results = Results()
    for entities, preprocessed_sentence, original_data in dataloader.generate_one_preprocessed_sample(preprocessor):
        sentence = Sentence(preprocessed_sentence.split(" "),
                            entities=entities, original_sentence=original_data,
                            preprocessed_sentence=preprocessed_sentence)

        batch = [sentence] * config_run["edition"]["number_of_edit_vector"]

        _, edit_traces = editor.edit(batch)

        results.add_edit_traces(edit_traces)

    best_results = results.compute_overall_best_n_results(n=10000)
    generated_sentences = results.sentences_without_entities_and_back_original_capitalization(best_results)

    dataset_dir = join(data.root, config_run["data_loader"]["dataset_foldername"])
    file_path = join(dataset_dir, "generated.txt")
    with open(file_path, "wb") as f:
        for sentence in generated_sentences:
            f.write(sentence.encode("utf8") + "\n")
