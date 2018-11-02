from training_run import MyEditTrainingRuns
from editor import GrllNeuralEditor
from config import GrllConfig
from data_loader import GrllDataLoader
from preprocess import GrllPreprocessor
from postprocess import GrllPostprocessor
from sentence import Sentence
from results import GrllResults
from metrics import GrllMetrics
from textmorph import data
from gtd.io import Workspace, sub_dirs
import re
from os.path import dirname, join
from writter import GrllWritter
from logger import logging_setup, config_run_logging_setup
import logging

# basic logging setup
logging_setup()

# Loading the Configs
configs = GrllConfig("textmorph/assess_results/config.json")

# Workspace setup
exps_workspace = Workspace(data.workspace.assess_results_runs)
if len(sub_dirs(exps_workspace.root)) == 0:
    exp_folder_name = "exp_" + str(0)
else:
    exp_num = max([int(re.search('(\d+)$', sub_dir_path).group(0)) for sub_dir_path in sub_dirs(exps_workspace.root)]) + 1
    exp_folder_name = "exp_"+str(exp_num)
exps_workspace.add_dir(exp_folder_name, exp_folder_name)
runs_workspace = Workspace(getattr(exps_workspace, exp_folder_name))

for idx, config_run in enumerate(configs):
    run_workspace = Workspace(getattr(runs_workspace, "run_" + str(idx)))
    config_run_file_handler = config_run_logging_setup(getattr(run_workspace, "stdout.txt"),
                                                       config_run["logger"]["console_level"],
                                                       config_run["logger"]["file_level"])
    GrllWritter.write(join(run_workspace.root, "config.txt"), config_run)

    exp = None
    if str(config_run["edit_model"]["exp_num"]).isdigit():  # If you don't want to load any model specify a non digit exp_num.
        logging.info("Loading the model from experiment #{}.".format(config_run["edit_model"]["exp_num"]))
        experiments = MyEditTrainingRuns()
        exp = experiments[config_run["edit_model"]["exp_num"]]

    if exp is None:
        logging.info("No model was loaded: no editor or word_vocab will be available in this config.".format(config_run["edit_model"]["exp_num"]))
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

    if config_run["data_loader"]["preprocess"]["show"]:  # write to file the output of the preprocessing phase
        file_path = join(run_workspace.root, config_run["data_loader"]["preprocess"]["filename"])
        GrllWritter.write_preprocessed_samples(file_path, dataloader.generate_one_preprocessed_sample())

    if editor is not None:
        logging.info("Editing the sentences and generating results.")
        results = GrllResults()
        postprocessor = GrllPostprocessor()
        for preprocessed_sentence, entities, original_sentence in dataloader.generate_one_preprocessed_sample():
            sentence = Sentence(preprocessed_sentence.split(" "),
                                entities=entities,
                                original_sentence=original_sentence,
                                preprocessed_sentence=preprocessed_sentence)

            batch = [sentence] * config_run["edition"]["number_of_edit_vector"]

            _, edit_traces = editor.edit(batch)
            filtered_candidates = postprocessor.postprocess_filter(edit_traces, min_number_of_token=4)
            results.add_candidates(filtered_candidates)

            if config_run["mode"] == "debug":
                break

        logging.info("{} filtered candidates were generated.".format(len(results)))
        results.sort()

        metrics = []
        for dataset_size in config_run["generation"]["dataset_sizes"]:
            logging.info("Attempting to generate dataset of size {}".format(dataset_size))
            generated_sentences, original_sentences = postprocessor.generate_n_sentences(results, n=dataset_size)
            logging.info("Dataset generated has size {}".format(len(generated_sentences)))

            bleu_score = GrllMetrics.compute_bleu_score(generated_sentences, original_sentences)
            perplexity = GrllMetrics.compute_perplexity([result["prob"] for result in results[:dataset_size]],
                                                        [len(result["sequence"]) for result in results[:dataset_size]])

            logging.info("BLEU score reported on this dataset: {}".format(bleu_score))
            logging.info("Perplexity reported on this dataset: {}".format(perplexity))
            metrics.append({"size": len(generated_sentences), "bleu_score": bleu_score, "perplexity": perplexity})

            file_path = join(run_workspace.root, "generated_"+str(len(generated_sentences))+".txt")
            GrllWritter.write(file_path, generated_sentences)

        file_path = join(run_workspace.root, "metrics.txt")
        GrllWritter.write_pretty_metrics(file_path, metrics)

    logging.getLogger().removeHandler(config_run_file_handler)
