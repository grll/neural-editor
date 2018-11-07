import codecs
from os.path import join
from textmorph import data
import logging
import pickle
import random



class ChurnDataLoader:
    """ Flexible dataloader for the neural editor generative model.

    Attributes:
        _data : raw data loaded.
        data_type : the type of the data loaded.
        _file_path : the file path of the file to load.
        _preprocessed_samples : a list of preprocessed samples.
    """

    def __init__(self, data_loader_config, preprocessor=None):
        """ Initialize a dataloader instance.

        Args:
            foldername (str): Name or Path corresponding to the folder.
            filename (str): Name of the file to load.
            data_type (str): Type of data to load among possible values are: ["csv"]
            preprocessor: A preprocessor to process the data must implement the preprocess method.
        """
        logging.info("Initializing the data loader.")
        dataset_dir = join(data.root, data_loader_config.dataset_foldername)
        self._file_path = join(dataset_dir, data_loader_config.dataset_filename)
        with codecs.open(self._file_path, "rb") as f:
            if data_loader_config.data_type == "csv":
                self._data = []
                self._headers = []
                for idx, line in enumerate(f):
                    line = line.decode('unicode_escape', 'ignore').encode("utf8").decode("utf8")  # solve encoding problems

                    line = line.replace("\n", "")
                    if idx == 0:
                        self._headers = line.split("\t")
                    else:
                        d = {}
                        for i, v in enumerate(line.split("\t")):
                            d[self._headers[i]] = v
                        self._data.append(d)
            else:
                raise NotImplementedError

        # Generate a testing and a training set:
        training_data, testing_data = self.split_training_testing(data_loader_config.training_size,
                                                                  data_loader_config.random_seed)
        self.write_csv("".join(self._file_path.split(".")[:-1]) + "_training.csv", training_data)
        self.write_csv("".join(self._file_path.split(".")[:-1]) + "_testing.csv", testing_data)

        self._data = training_data  # set the internal data on which the loader is relying to be only the training_data
        self.data_type = data_loader_config.data_type
        self.preprocessor = preprocessor
        self._preprocessed_samples = []

    def split_training_testing(self, training_size=200, seed=7):
        """ Split the dataset into random training/testing respecting 50% population for each class in the training set

        Args:
            training_size: the size of the wanted training set.
            seed: the random seed to use.

        Returns:
            training_data, testing_data: a tuple of 2 list of dict.
        """
        random.seed(seed)

        churny_samples = [d for d in self._data if int(d[u'label']) == 1]
        non_churny_samples = [d for d in self._data if int(d[u'label']) == 0]

        random.shuffle(churny_samples)
        random.shuffle(non_churny_samples)

        training_data = churny_samples[:training_size/2] + non_churny_samples[:training_size/2]
        testing_data = churny_samples[training_size/2:] + non_churny_samples[training_size/2:]

        random.shuffle(training_data)
        random.shuffle(testing_data)

        return training_data, testing_data

    def write_csv(self, path, data):
        """ Write some data in CSV format from a list of dict """
        with open(path, "wb") as f:
            f.write("\t".join(self._headers).encode("utf8")+"\n")
            for d in data:
                f.write("\t".join([d[header] for header in self._headers]).encode("utf8")+"\n")

    def generate_one_sample(self):
        """ Yield data samples one by one. """
        if self.data_type == "csv":
            for data in self._data:
                logging.debug("Yielding dict data: {}".format(data))
                yield data  # yield a dict with the csv headers
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
                    preprocessed_sentence.encode("utf8"), entities, data))
                yield preprocessed_sentence, entities, data
        else:
            self._preprocessed_samples = []
            for data in self.generate_one_sample():
                if self.data_type == "csv":
                    preprocessed_sentence, entities = self.preprocessor.preprocess(data[u"text_org"])
                    logging.debug("Yielding and Saving Following Data:\npreprocessed_sentence:{}\nentities:{}\ndata:{}".format(
                        preprocessed_sentence.encode("utf8"), entities, data.encode("utf8")))
                    self._preprocessed_samples.append((preprocessed_sentence, entities, data))
                    yield preprocessed_sentence, entities, data
                else:
                    raise NotImplementedError

    def write_preprocessed_samples(self, file_path):
        """ Write the preprocessing output to file. """
        with open(file_path, "wb") as f:
            if len(self._preprocessed_samples) == len(self._data):
                for preprocessed_sentence, entities, data in self._preprocessed_samples:
                    f.write(data[u"text_org"].encode("utf8") + "\n")
                    f.write("> " + preprocessed_sentence.encode("utf8") + "\n")
                    f.write("> " + str(entities) + "\n\n")
            else:
                raise Exception("You need to preprocess the samples before outputting them to file.")

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
            if self.data_type == "csv":
                preprocessed_sentence, entities = self.preprocessor.preprocess(data[u"text_org"])
                logging.debug("Saving Following Preprocessed Data:\npreprocessed_sentence:{}\nentities:{}\ndata:{}".format(preprocessed_sentence.encode("utf8"), entities, data))
                self._preprocessed_samples.append((preprocessed_sentence, entities, data))
            else:
                raise NotImplementedError

        with open(self._file_path + "_preprocessed.pickle", "wb") as f:
            pickle.dump(self._preprocessed_samples, f)
        logging.info("Preprocessing done and preprocessed samples saved in `{}`".format(self._file_path + "_preprocessed.pickle"))
        return True
