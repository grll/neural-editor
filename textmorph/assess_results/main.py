from training_run import MyEditTrainingRuns
from editor import GrllNeuralEditor
from config import GrllConfig
from data_loader import GrllDataLoader
from preprocess import GrllPreprocessor
from postprocess import GrllPostprocessor
from metrics import GrllMetrics
from gtd.io import Workspace
from workspace import setup_exps_workspace
from os.path import join
from writter import GrllWritter
from logger import logging_setup, config_run_logging_setup, handle_exception
import sys
import logging

#TODO parse arg to the config file with default to default

# basic logging setup
logging_setup()
sys.excepthook = handle_exception

# Loading the Configs
configs = GrllConfig()

# Workspace setup
runs_workspace = setup_exps_workspace("assess_results_runs")

for idx, config_run in enumerate(configs):
    # 0. Setup the Config Run
    config_ = config_run.obj
    run_workspace = Workspace(getattr(runs_workspace, "run_" + str(idx)))  # setup the workspace for the given run
    config_run_file_handler = config_run_logging_setup(getattr(run_workspace, "stdout.txt"),  # setup the logs for the given config_run
                                                       config_.logger.console_level,
                                                       config_.logger.file_level)
    config_run.write_to_file(join(run_workspace.root, "config.txt")) # write the current run_config to config.txt

    # 1. Loading the model
    edit_model = MyEditTrainingRuns().load_edit_model(config_.edit_model.exp_num)  # load the edit_model.
    editor = GrllNeuralEditor(edit_model)  # create custom editor with random edition vectors function.
    word_vocab = edit_model.train_decoder.word_vocab  # load the word vocabulary used with this specific edit_model.

    # 2. Preprocessing & data loading
    preprocessor = GrllPreprocessor(word_vocab=word_vocab, lang=config_.lang)
    dataloader = GrllDataLoader(config_.data_loader.dataset_foldername,
                                config_.data_loader.dataset_filename,
                                config_.data_loader.data_type,
                                preprocessor)
    dataloader.preprocess_all(config_.data_loader.preprocess.force)  # preprocess all or retrieve from file
    if config_.data_loader.preprocess.show:  # write to file the output of the preprocessing phase
        file_path = join(run_workspace.root, config_.data_loader.preprocess.filename)
        GrllWritter.write_preprocessed_samples(file_path, dataloader.generate_one_preprocessed_sample())

    # 3. Running the model with preprocessed data
    postprocessor = GrllPostprocessor()
    results = editor.run_model(config_.edit_model, dataloader, postprocessor)
    results.sort()

    # 4. Generate the Datasets & Metrics
    metrics = []
    for dataset_size in config_.generation.dataset_sizes:
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
