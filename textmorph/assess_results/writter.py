import logging
import yaml
from prettytable import PrettyTable

class GrllWritter():
    """ Different writting function to file. """

    @classmethod
    def write_preprocessed_samples(cls, path, packed_samples):
        """ Write the preprocessed_samples to the file specified in path

        Args:
            path: path where to write the preprocessed samples
            packed_samples: a generator of tuples of (original_sentence, preprocessed_sentente, entities)
        """
        logging.info("Writting the preprocessed samples to file.")
        with open(path, "wb") as f:
            for preprocessed_sentence, entities, original_sentence in packed_samples:
                cls._write_preprocessed_sample(f, original_sentence, preprocessed_sentence, entities)

    @classmethod
    def _write_preprocessed_sample(cls, f, original_sentence, preprocessed_sentence, entities):
        """ Format and write the preprocessed sample in the file.

        Args:
            f: file handler
            original_sentence: original sentence
            preprocessed_sentence: preprocessed sentence
            entities: dict of entities
        """
        f.write(original_sentence.encode("utf8") + "\n")
        f.write("> " + preprocessed_sentence.encode("utf8") + "\n")
        f.write("> " + str(entities) + "\n\n")

    @classmethod
    def write(cls, path, content):
        if type(content) is dict:
            with open(path, "wb") as f:
                f.write(yaml.safe_dump(content))
        elif type(content) is list:
            if type(content[0]) is unicode:
                with open(path, "wb") as f:
                    for line in content:
                        f.write(line.encode("utf8") + "\n")
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    @classmethod
    def write_pretty_metrics(cls, path, metrics):
        x = PrettyTable()
        x.field_names = list(metrics[0].keys())
        for metric in metrics:
            x.add_row(list(metric.values()))
        with open(path, "wb") as f:
            f.write(x.get_string())
