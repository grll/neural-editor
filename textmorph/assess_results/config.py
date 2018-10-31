import json
from os.path import join, dirname
from logger import GrllLogger

logger = GrllLogger(name="assess_results", level="DEBUG")

class GrllConfig:
    """ Handle multiple config when multiple runs are defined.

    Attributes:
         _data: Raw JSON data of the config file.
         _config: Global config used has default value to generate single run config.
    """

    def __init__(self, config_file_path="default"):
        """ Initialize the config object by loading the config JSON file.

        Args:
            config_file_path: Path toward the config.json file.
        """
        logger.info("Loading the configs.")
        if config_file_path == "default":
            config_file_path = join(dirname("__FILE__"), "config.json")

        if config_file_path is not "nofile":
            with open(config_file_path, "r") as f:
                self._data = json.load(f)

            self._config = {}
            for k,v in self._data["global_config"].items():
                self._config[k] = v  # Set global config attributes that are overridden in each runs.
        logger.info("{} config(s) successfully loaded from {}.".format(len(self._data["runs_config"]), config_file_path))

    def __iter__(self):
        """ Yield individual config_run one by one merging the individual config with the global config already in place

        Yields:
            run_config : A specific config for one run.
        """
        for run_config in self._data["runs_config"]:
            run_config = self.merge(run_config, self._config)
            logger.info("The following config has been loaded: \n {}".format(run_config))
            yield run_config

    def __len__(self):
        """ Return the number of run_configs """
        return len(self._data["runs_config"])

    def __getitem__(self, item):
        """ Get a specific run using [item] """
        return self.merge(self._data["runs_config"][i], self._config)

    def merge(self, truth_dict, completing_dict):
        """
        Recursive function that merge two dictionary and add every key / value from completing dict if and only if
        they don't already exists in the truth dictionary.

        Args:
            truth_dict: A dictionary ground_truth dictionary which values should not be altered.
            completing_dict: A dictionary to complete the ground_truth if the values are not set in the first place.

        Returns:

        """
        for k, v in completing_dict.items():
            if k not in truth_dict:
                truth_dict[k] = v
            else:
                if type(v) == dict and type(truth_dict[k]) == dict:
                    self.merge(truth_dict[k], v)
        return truth_dict
