import unittest
from undeepvo.utils import TrainingProcessHandler


class TestTrainingProcessHandler(unittest.TestCase):
    def test_training_process_handler(self):
        handler = TrainingProcessHandler(enable_mlflow=False)
        handler.setup_handler("test", None)
        handler.start_callback(1, 1)
        handler.epoch_callback({})
        handler.finish_callback({})
