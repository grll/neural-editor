from training_run import MyEditTrainingRuns
from editor import GrllNeuralEditor
from config import GrllConfig
from data_loader import GrllDataLoader
from preprocess import GrllPreprocessor
from postprocess import GrllPostprocessor
from metrics import GrllMetrics
from textmorph import data
from gtd.io import Workspace, sub_dirs
import re
from os.path import dirname, join
from writter import GrllWritter
from logger import logging_setup, config_run_logging_setup, handle_exception
import sys
import logging

#parse arg to the config file with default to default

# basic logging setup
logging_setup()
sys.excepthook = handle_exception

# Loading the Configs
configs = GrllConfig()

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
    # 0. Setup the Config Run
    run_workspace = Workspace(getattr(runs_workspace, "run_" + str(idx)))  # setup the workspace for the given run
    config_run_file_handler = config_run_logging_setup(getattr(run_workspace, "stdout.txt"),  # setup the logs for the given config_run
                                                       config_run["logger"]["console_level"],
                                                       config_run["logger"]["file_level"])
    config_run.write_to_file(join(run_workspace.root, "config.txt")) # write the current run_config to config.txt

    # 1. Loading the model
    edit_model = MyEditTrainingRuns().load_edit_model(config_run["edit_model"]["exp_num"])  # load the edit_model.
    editor = GrllNeuralEditor(edit_model)  # create custom editor with random edition vectors function.
    word_vocab = edit_model.train_decoder.word_vocab  # load the word vocabulary used with this specific edit_model.

    # 2. Preprocessing & data loading
    preprocessor = GrllPreprocessor(word_vocab=word_vocab, lang=config_run["lang"])
    dataloader = GrllDataLoader(config_run["data_loader"]["dataset_foldername"],
                                config_run["data_loader"]["dataset_filename"],
                                config_run["data_loader"]["data_type"],
                                preprocessor)
    dataloader.preprocess_all(config_run["data_loader"]["preprocess"]["force"])  # preprocess all or retrieve from file
    if config_run["data_loader"]["preprocess"]["show"]:  # write to file the output of the preprocessing phase
        file_path = join(run_workspace.root, config_run["data_loader"]["preprocess"]["filename"])
        GrllWritter.write_preprocessed_samples(file_path, dataloader.generate_one_preprocessed_sample())

    # 3. Running the model with preprocessed data
    postprocessor = GrllPostprocessor()
    results = editor.run_model(config_run, dataloader, postprocessor)
    results.sort()

    # 4. Generate the Datasets & Metrics
    metrics = []
    for dataset_size in config_run["generation"]["dataset_sizes"]:
        generated_sentences, original_sentences = postprocessor.generate_n_sentences(results, n=dataset_size)

        bleu_score = GrllMetrics.compute_bleu_score(generated_sentences, original_sentences)
        perplexity = GrllMetrics.compute_perplexity([result["prob"] for result in results[:dataset_size]],
                                                    [len(result["sequence"]) for result in results[:dataset_size]])

        metrics.append({"size": len(generated_sentences), "bleu_score": bleu_score, "perplexity": perplexity})

        file_path = join(run_workspace.root, "generated_"+str(len(generated_sentences))+".txt")
        GrllWritter.write(file_path, generated_sentences)

    file_path = join(run_workspace.root, "metrics.txt")
    GrllWritter.write_pretty_metrics(file_path, metrics)

    logging.getLogger().removeHandler(config_run_file_handler)
