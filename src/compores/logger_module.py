# logger_module.py
import logging
import sys


class CompoResLogger:
    # Class-level dictionary to keep track of log files and their associated file handlers
    _file_handlers = {}

    # Create a generic logger
    def __init__(
            self, log_name, log_file=None, console_log_level=logging.INFO, log_file_level=logging.INFO, filemode='w'
    ):
        """Basic logger setup
        :param log_name: Name of the logger, can be __name__
        :param log_file: Name of the log file, can be None if no log file is needed
        :param console_log_level: Level of the log output on the console, e.g. logging.DEBUG
        :param log_file_level: Level of the log output in the file, e.g. logging.INFO
        :param filemode: Mode of the log file, e.g. 'w' or 'a'
        """
        # Create the logger object
        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(logging.DEBUG)

        # Define the log output format
        self.formatter = logging.Formatter('%(levelname)s\t%(asctime)s\t%(name)s\t%(message)s\n')

        self.log_file = log_file
        self.log_file_level = log_file_level
        self.filemode = filemode

        # Check if handlers already exist and add them only if they don't
        if not self.logger.handlers:

            # Create the console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(console_log_level)
            console_handler.setFormatter(self.formatter)
            self.logger.addHandler(console_handler)

    def _add_file_handler(self, log_file, log_file_level, filemode):
        file_handler = logging.FileHandler(filename=log_file, mode=filemode)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(self.formatter)
        self.logger.addHandler(file_handler)
        #
        # # Add log file and associated file handler to class-level dictionary
        # CompoResLogger._file_handlers[log_file] = file_handler

    def cleanup_logger_file_handlers(self):
        handlers = self.logger.handlers[:]
        for handler in handlers:
            if isinstance(handler, logging.FileHandler):
                handler.close()
                self.logger.removeHandler(handler)

    def update_logger_file_handler(self, new_log_file):
        if new_log_file != self.log_file and new_log_file not in CompoResLogger._file_handlers:
            # Remove all existing file handlers
            self.cleanup_logger_file_handlers()
            # Add the new file handler
            self._add_file_handler(new_log_file, self.log_file_level, self.filemode)
            self.log_file = new_log_file

    def get_logger(self):
        return self.logger
