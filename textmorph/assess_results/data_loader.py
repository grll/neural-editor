import codecs
from os.path import join
from textmorph import data
import logging
import pickle


class GrllDataLoader:
    """ Flexible dataloader for the neural editor generative model.

    Attributes:
        _data : raw data loaded.
        data_type : the type of the data loaded.
        _file_path : the file path of the file to load.
        _preprocessed_samples : a list of preprocessed samples.
    """

    def __init__(self, foldername, filename, data_type, preprocessor=None):
        """ Initialize a dataloader instance.

        Args:
            foldername (str): Name or Path corresponding to the folder.
            filename (str): Name of the file to load.
            data_type (str): Type of data to load among possible values are: ["one_line_one_sentence"]
            preprocessor: A preprocessor to process the data must implement the preprocess method.
        """
        logging.info("Initializing the data loader.")
        dataset_dir = join(data.root, foldername)
        self._file_path = join(dataset_dir, filename)
        with codecs.open(self._file_path, "rb", encoding="utf-8") as f:
            if data_type == "one_line_one_sentence":
                self._data = []
                for line in f:
                    self._data.append(line.replace("\n", ""))
            else:
                raise NotImplementedError
        self.data_type = data_type
        self.preprocessor = preprocessor
        self._preprocessed_samples = []

    def generate_one_sample(self):
        """ Yield data samples one by one. """
        if self.data_type == "one_line_one_sentence":
            for data in self._data:
                logging.debug("Yielding sentence: {}".format(data.encode("utf8")))
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
        if len(self._preprocessed_samples) == len(self._data):
            for preprocessed_sentence, entities, data in self._preprocessed_samples:
                logging.debug("Yielding Following Data from memory:\npreprocessed_sentence:{}\nentities:{}\ndata:{}".format(
                    preprocessed_sentence.encode("utf8"), entities, data.encode("utf8")))
                yield preprocessed_sentence, entities, data
        else:
            self._preprocessed_samples = []
            for data in self.generate_one_sample():
                if self.data_type == "one_line_one_sentence":
                    preprocessed_sentence, entities = self.preprocessor.preprocess(data)
                    logging.debug("Yielding and Saving Following Data:\npreprocessed_sentence:{}\nentities:{}\ndata:{}".format(
                        preprocessed_sentence.encode("utf8"), entities, data.encode("utf8")))
                    self._preprocessed_samples.append((preprocessed_sentence, entities, data))
                    yield preprocessed_sentence, entities, data
                else:
                    raise NotImplementedError

    def preprocess_all(self, force_preprocessing):
        """ Preprocess all the data or load them from pickle file.

        Args:
            force_preprocessing: Bool weather to force preprocessing or autoload from pickle file if exist.
        """
        if not force_preprocessing:
            try:
                with open(self._file_path + "_preprocessed.pickle", "rb") as f:
                    self._preprocessed_samples = pickle.load(f)
                    logging.info("Preprocessed sample pickle file successfully loaded.")
                    return True
            except IOError:
                logging.info("Couldn't find a preprocessed pickle file.")

        logging.info("Preprocessing all the data and saving a new pickle.")
        for data in self.generate_one_sample():
            if self.data_type == "one_line_one_sentence":
                preprocessed_sentence, entities = self.preprocessor.preprocess(data)
                logging.debug("Saving Following Preprocessed Data:\npreprocessed_sentence:{}\nentities:{}\ndata:{}".format(preprocessed_sentence.encode("utf8"), entities, data.encode("utf8")))
                self._preprocessed_samples.append((preprocessed_sentence, entities, data))
            else:
                raise NotImplementedError

        with open(self._file_path + "_preprocessed.pickle", "wb") as f:
            pickle.dump(self._preprocessed_samples, f)
        logging.info("Preprocessing done and preprocessed samples saved in `{}`".format(self._file_path + "_preprocessed.pickle"))
        return True