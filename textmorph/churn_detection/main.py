from textmorph.assess_results.training_run import MyEditTrainingRuns
from editor import ChurnNeuralEditor
from textmorph.assess_results.config import GrllConfig
from data_loader import ChurnDataLoader
from preprocess import ChurnPreprocessor
from postprocess import ChurnPostProcessor
from textmorph.assess_results.metrics import GrllMetrics
from textmorph.assess_results.workspace import setup_exps_workspace
from textmorph.assess_results.writter import GrllWritter
from textmorph.assess_results.logger import logging_setup, config_run_logging_setup, handle_exception

from gtd.io import Workspace

from os.path import join, dirname, abspath
import sys
import logging

#TODO parse arg to the config file with default to default

# basic logging setup
logging_setup()
sys.excepthook = handle_exception

# Loading the Configs
configs = GrllConfig(join(dirname(abspath(__file__)), "configs/default.yml"))

# Workspace setup
runs_workspace = setup_exps_workspace("churn_augmentation_runs")

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
    editor = ChurnNeuralEditor(edit_model)  # create custom editor with random edition vectors function.
    word_vocab = edit_model.train_decoder.word_vocab  # load the word vocabulary used with this specific edit_model.

    # 2. Preprocessing & data loading
    preprocessor = ChurnPreprocessor(word_vocab=word_vocab, lang=config_.lang)
    dataloader = ChurnDataLoader(config_.data_loader, preprocessor)
    dataloader.preprocess_all(config_.data_loader.preprocess.force)  # preprocess all or retrieve from file
    if config_.data_loader.preprocess.show:  # write to file the output of the preprocessing phase
        file_path = join(run_workspace.root, config_.data_loader.preprocess.filename)
        dataloader.write_preprocessed_samples(file_path)

    # 3. Running the model with preprocessed data
    postprocessor = ChurnPostProcessor()
    results = editor.run_model(config_.edit_model, dataloader, postprocessor)
    results.sort()

    # 4. Generate the Datasets & Metrics
    metrics = []
    for dataset_size in config_.generation.dataset_sizes:
        generated_sentences, original_data = postprocessor.generate_n_sentences(results, n=dataset_size)

        bleu_score = GrllMetrics.compute_bleu_score(generated_sentences, [d[u"text_org"] for d in original_data])
        perplexity = GrllMetrics.compute_perplexity([result["prob"] for result in results[:dataset_size]],
                                                    [len(result["sequence"]) for result in results[:dataset_size]])

        metrics.append({"size": len(generated_sentences), "bleu_score": bleu_score, "perplexity": perplexity})

        file_path = join(run_workspace.root, "generated_"+str(len(generated_sentences))+".txt")
        GrllWritter.write(file_path, generated_sentences)

    file_path = join(run_workspace.root, "metrics.txt")
    GrllWritter.write_pretty_metrics(file_path, metrics)

    logging.getLogger().removeHandler(config_run_file_handler)
