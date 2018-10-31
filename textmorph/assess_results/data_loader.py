import codecs
from os.path import join
from textmorph import data
import logging

logger = logging.getLogger("assess_results")
logger.propagate = False


class GrllDataLoader:
    """ Flexible dataloader for the neural editor generative model.

    Attributes:
        _data : raw data loaded.
        data_type : the type of the data loaded.
    """

    def __init__(self, foldername, filename, data_type, preprocessor=None):
        """ Initialize a dataloader instance.

        Args:
            foldername (str): Name or Path corresponding to the folder.
            filename (str): Name of the file to load.
            data_type (str): Type of data to load among possible values are: ["one_line_one_sentence"]
            preprocessor: A preprocessor to process the data must implement the preprocess method.
        """
        logger.info("Initializing the data loader.")
        dataset_dir = join(data.root, foldername)
        file_path = join(dataset_dir, filename)
        with codecs.open(file_path, "rb", encoding="utf-8") as f:
            if data_type == "one_line_one_sentence":
                self._data = []
                for line in f:
                    self._data.append(line.replace("\n", ""))
            else:
                raise NotImplementedError
        self.data_type = data_type
        self.preprocessor = preprocessor

    def generate_one_sample(self):
        """ Yield data samples one by one. """
        if self.data_type == "one_line_one_sentence":
            for data in self._data:
                logger.debug("Yielding sentence: {}".format(data.encode("utf8")))
                yield data
        else:
            raise NotImplementedError

    def generate_one_preprocessed_sample(self):
        """ Yield a preprocessed sample from the dataset.

        Args:
            preprocessor: the preprocessor to use must implement the following method: preprocess

        Yields:
            preprocessed_sample: yields the preprocessed sample one by one.
        """
        for data in self.generate_one_sample():
            if self.data_type == "one_line_one_sentence":
                preprocessed_sentence, entities = self.preprocessor.preprocess(data)
                logger.debug("Yielding Following Data:\npreprocessed_sentence:{}\nentities:{}\ndata:{}".format(preprocessed_sentence.encode("utf8"), entities, data.encode("utf8")))
                yield preprocessed_sentence, entities, data
            else:
                raise NotImplementedError
