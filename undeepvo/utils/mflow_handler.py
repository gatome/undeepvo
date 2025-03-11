import os
import mlflow
import mlflow.exceptions

# Use the ngrok URL instead of localhost
DEFAULT_HOST_URI = "https://7306-85-191-70-247.ngrok-free.app"
DEFAULT_EXPERIMENT_NAME = "undeepvo"

class MlFlowHandler(object):
    def __init__(self, experiment_name=DEFAULT_EXPERIMENT_NAME, host_uri=DEFAULT_HOST_URI, mlflow_tags={}, mlflow_parameters={}):
        self._enable_mlflow = True
        self._experiment_name = experiment_name
        self._mlflow_tags = mlflow_tags
        self._mlflow_parameters = mlflow_parameters

        try:
            mlflow.set_tracking_uri(host_uri)
            self._mlflow_client = mlflow.tracking.MlflowClient()
            mlflow.set_experiment(self._experiment_name)
        except Exception as e:
            print(f"[ERROR] Failed to connect to MLflow server: {e}")
            self._enable_mlflow = False

    def start_callback(self, parameters):
        if not self._enable_mlflow:
            return
        try:
            if mlflow.active_run() is not None:
                mlflow.end_run()
            mlflow.start_run()
            mlflow.set_tags(self._mlflow_tags)
            mlflow.log_params(parameters)
            mlflow.log_params(self._mlflow_parameters)
        except mlflow.exceptions.MlflowException as msg:
            self._enable_mlflow = False
            print(f"[WARNING] - [StartCallback] {msg}")

    def finish_callback(self):
        if not self._enable_mlflow:
            return
        try:
            mlflow.end_run()
        except mlflow.exceptions.MlflowException as msg:
            self._enable_mlflow = False
            print(f"[WARNING] - [FinishCallback] {msg}")

    def epoch_callback(self, metrics, current_epoch=0, artifacts=None):
        if not self._enable_mlflow:
            return
        try:
            metrics["epoch"] = current_epoch
            mlflow.log_metrics(metrics, current_epoch)
            if artifacts:
                for artifact in artifacts:
                    mlflow.log_artifact(artifact)
                    os.remove(artifact)  # Clean up artifacts after logging
        except mlflow.exceptions.MlflowException as msg:
            self._enable_mlflow = False
            print(f"[WARNING] - [EpochCallback] {msg}")
