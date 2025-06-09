# test_my_logger.py

import os
import shutil

import pytest
from compores.logger_module import CompoResLogger


@pytest.fixture
def setup_logger():
    # Create an instance of MyLogger with an initial log file
    log_name = 'test_logger'
    initial_log_file = 'test_initial.log'
    logger_instance = CompoResLogger(log_name=log_name, log_file=initial_log_file)
    logger = logger_instance.get_logger()
    return logger_instance, initial_log_file, logger


class TestLogger:

    @pytest.fixture(scope="function")
    def logger_instance(self, tmp_path):
        log_name = 'test_logger'
        log_file = tmp_path / 'test_log.log'
        logger_instance = CompoResLogger(log_name=log_name, log_file=str(log_file))
        yield logger_instance, log_file
        self.close_handlers(logger_instance)

    @staticmethod
    def close_handlers(logger_instance):
        handlers = logger_instance.get_logger().handlers[:]
        for handler in handlers:
            handler.close()
            logger_instance.get_logger().removeHandler(handler)

    def test_update_logger_file_name(self, logger_instance):
        logger_instance, log_file = logger_instance
        os.makedirs('tmp', exist_ok=True)
        new_log_file = os.path.join('tmp', 'updated_log.log')

        logger_instance.update_logger_file_handler(str(new_log_file))

        logger_instance.get_logger().info('This is an info message in the updated log file.')

        self.close_handlers(logger_instance)

        # assert new_log_file.exists()
        with open(new_log_file, 'r') as file:
            log_contents = file.read()
            assert 'This is an info message in the updated log file.' in log_contents

        if os.path.exists("tmp"):
            shutil.rmtree("tmp")

        if os.path.exists("logs"):
            shutil.rmtree("logs")
