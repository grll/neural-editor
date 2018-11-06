from os.path import realpath, join, dirname
from textmorph.edit_model.training_run import EditTrainingRuns, EditTrainingRun
from textmorph import data
import logging

class MyEditTrainingRuns(EditTrainingRuns):
    """ Inherit from EditTraining Runs used here to prevent loading of the whole dataset file. """

    def __init__(self):
        data_dir = data.workspace.edit_runs
        src_dir = dirname(dirname(dirname(realpath('__file__'))))  # root of the repo
        super(EditTrainingRuns, self).__init__(data_dir, src_dir, MyEditTrainingRun, check_commit=False)

    def load_edit_model(self, exp_num):
        """ Load and return an edit_model from an exp_num.

        Args:
            exp_num: a number representing a training experiment from the neural editor model.

        Returns:
            edit_model: exp.editor coming from the experiment.
        """
        logging.info("Loading the model from experiment #{}.".format(config_run["edit_model"]["exp_num"]))
        return self[exp_num].editor

class MyEditTrainingRun(EditTrainingRun):
    """ The actual modification of the loading appears ot be in this class that is used in the class just above. """

    def __init__(self, config, save_dir):
        super(EditTrainingRun, self).__init__(config, save_dir)

        # extra dir for storing TrainStates where NaN was encountered
        self.workspace.add_dir('nan_checkpoints', 'nan_checkpoints')

        # reload train state (includes model)
        checkpoints_dir = self.workspace.checkpoints
        ckpt_num = self._get_latest_checkpoint_number(checkpoints_dir)
        if ckpt_num is None:
            print 'No checkpoint to reload. Initializing fresh.'
            self._train_state = self._initialize_train_state(config)
        else:
            print 'Reloaded checkpoint #{}'.format(ckpt_num)
            self._train_state = self._reload_train_state(checkpoints_dir, ckpt_num, config)
