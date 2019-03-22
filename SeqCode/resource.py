# -*- coding: utf-8 -*-
import json
import os
import subprocess
import sys
from datetime import datetime

import logzero
from logzero import logger


class Resource(object):
    """
    Helper class for managing the experiment.
    """

    def __init__(self, args, train=True, log_filename='train.log', output_dir=None):
        self.args = args  # argparse object
        self.logger = logger
        self.start_time = datetime.today()
        self.config = None  # only used for the inference process

        if train:  # for training
            self.output_dir = self._return_output_dir()
            self.create_output_dir()
        else:  # for inference
            assert output_dir is not None
            self.output_dir = output_dir

        log_name = os.path.join(self.output_dir, log_filename)
        logzero.logfile(log_name)
        self.log_name = log_name
        self.logger.info('Log filename: [{}]'.format(log_name))

    def _return_output_dir(self):
        dir_name = 'log'
        dir_name += '_optim_{}'.format(self.args.optimizer)
        dir_name += '_lr_{}'.format(self.args.lr)
        dir_name += '_batch_{}'.format(self.args.batch_size)

        output_dir = os.path.abspath(os.path.join(self.args.out, dir_name))
        return output_dir

    def create_output_dir(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            self.logger.info('Output Dir is created at [{}]'.format(self.output_dir))
        else:
            self.logger.warn('Output Dir [{}] already exists'.format(self.output_dir))
            self.logger.warn('Overwriting log (and other important files) is not allowed. Exit...')
            # exit(1)

    def dump_git_info(self):
        """
        returns git commit id, diffs from the latest commit
        """
        if os.system('git rev-parse 2> /dev/null > /dev/null') == 0:
            self.logger.info('Git repository is found. Dumping logs & diffs...')
            git_log = '\n'.join(
                subprocess.check_output('git log -1 --abbrev-commit', shell=True).decode('utf8').split('\n'))
            self.logger.info(git_log)
            git_diff = subprocess.check_output('git diff', shell=True).decode('utf8')
            self.logger.info(git_diff)
        else:
            self.logger.warn('Git repository is not found. DO NOT run experiment unless you have git repo.')

    def dump_command_info(self):
        """
        returns command line arguments / command path / name of the node
        """
        self.logger.info('Command name: {}'.format(' '.join(sys.argv)))
        self.logger.info('Command is executed at: [{}]'.format(os.getcwd()))
        self.logger.info('Current program is running at: [{}]'.format(os.uname().nodename))

    def dump_python_info(self):
        """
        returns python version info
        """
        self.logger.info('Python Version: [{}]'.format(sys.version.replace('\n', '')))

    def save_config_file(self):
        """
        save argparse object into config.json
        config.json is used during the inference
        """
        with open(os.path.join(self.output_dir, 'config.json'), 'w') as fo:
            dumped_config = json.dumps(vars(self.args), sort_keys=True, indent=4)
            fo.write(dumped_config)
            self.logger.info('Hyper-Parameters: {}'.format(dumped_config))

    def load_config(self):
        """
        load config.json and recover hyperparameters that are used during the training
        """
        model_directory = os.path.dirname(self.args.model)
        config_path = os.path.join(model_directory, 'config.json')
        self.config = json.load(open(config_path, 'r'))
        self.logger.info('Loaded config from {}'.format(config_path))

    def get_model_path(self):
        path = os.path.join(self.args.model)
        assert os.path.exists(path), 'Model file does not Exist...'
        return path
