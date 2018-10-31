from training_run import MyEditTrainingRuns
from editor import GrllNeuralEditor
from config import GrllConfig
from data_loader import GrllDataLoader
from preprocess import GrllPreprocessor
from sentence import Sentence
from results import Results
from textmorph import data
from os.path import dirname, join
from logger import GrllLogger


logger = GrllLogger(level="DEBUG")

configs = GrllConfig("textmorph/assess_results/config.json")

for config_run in configs:

    exp = None
    if str(config_run["edit_model"]["exp_num"]).isdigit():  # If you don't want to load any model specify a non digit exp_num.
        logger.info("Loading the model from experiment #{}.".format(config_run["edit_model"]["exp_num"]))
        experiments = MyEditTrainingRuns()
        exp = experiments[config_run["edit_model"]["exp_num"]]

    if exp is None:
        logger.info("No model was loaded: no editor or word_vocab will be available in this config.".format(config_run["edit_model"]["exp_num"]))
        editor = None
        word_vocab = None
    else:
        editor = GrllNeuralEditor(exp.editor)  # create custom editor with random edition.
        word_vocab = exp.editor.train_decoder.word_vocab  # word vocabulary used during training of the editor.

    preprocessor = GrllPreprocessor(word_vocab=word_vocab, lang=config_run["lang"])

    dataloader = GrllDataLoader(config_run["data_loader"]["dataset_foldername"],
                                config_run["data_loader"]["dataset_filename"],
                                config_run["data_loader"]["data_type"],
                                preprocessor)

    dataloader.preprocess_all(config_run["data_loader"]["force_preprocessing"])
    if config_run["data_loader"]["preprocess"]["show"]:
        dataset_dir = join(data.root, config_run["data_loader"]["dataset_foldername"])
        file_path = join(dataset_dir, config_run["data_loader"]["preprocess"]["filename"])
        for preprocessed_sentence, entities, original_sentence in dataloader.generate_one_preprocessed_sample():
            logger.log_preprocessed_sentences(file_path, original_sentence.encode("utf8"), preprocessed_sentence.encode("utf8"), str(entities))
            f.write(original_sentence.encode("utf8") + "\n")
            f.write("> " + preprocessed_sentence.encode("utf8") + "\n")
            f.write("> " + str(entities) + "\n\n")

    if editor is not None:
        results = Results()
        for preprocessed_sentence, entities, original_sentence in dataloader.generate_one_preprocessed_sample():
            sentence = Sentence(preprocessed_sentence.split(" "),
                                entities=entities,
                                original_sentence=original_sentence,
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
