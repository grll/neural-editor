import logging

class GrllLogger():
    """ Define a custom logger to handle input/output. """
    def __init__(self, name="default", level="DEBUG"):
        self._logger = logging.getLogger(name)

        levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }

        self._logger.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(levels[level])
        console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
        console_handler.setFormatter(console_formatter)

        self._logger.addHandler(console_handler)

    def debug(self, msg, *args, **kwargs):
        self._logger.debug(msg, args, kwargs)

    def info(self, msg, *args, **kwargs):
        self._logger.info(msg, args, kwargs)

    def warning(self, msg, *args, **kwargs):
        self._logger.warning(msg, args, kwargs)

    def error(self, msg, *args, **kwargs):
        self._logger.error(msg, args, kwargs)

    def critical(self, msg, *args, **kwargs):
        self._logger.critical(msg, args, kwargs)

