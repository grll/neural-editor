import logging

logFormatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
rootLogger = logging.getLogger()
consoleHandler = logging.StreamHandler()


# Logging setup
def logging_setup():
    rootLogger.setLevel(logging.DEBUG)
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)


# Config run logging setup
def config_run_logging_setup(file_path, console_level, file_level):
    config_run_file_handler = logging.FileHandler(file_path)  # set the file logger
    config_run_file_handler.setFormatter(logFormatter)
    rootLogger.addHandler(config_run_file_handler)
    consoleHandler.setLevel(getattr(logging, console_level))
    config_run_file_handler.setLevel(getattr(logging, file_level))
    return config_run_file_handler
