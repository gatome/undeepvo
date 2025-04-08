import os
import warnings
import torch
import random
import kornia
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim

from undeepvo.models import UnDeepVO
from undeepvo.criterion import UnsupervisedCriterion
from undeepvo.problems import UnsupervisedDatasetManager, UnsupervisedDepthProblem
from undeepvo.utils import OptimizerManager, TrainingProcessHandler

# Set the Rosario dataset path
MAIN_DIR = '/content/drive/MyDrive/rosario_dataset'

# Sequence ID (Rosario only has one we're using for now)
sequence = '01'
image_count = len(os.listdir(os.path.join(MAIN_DIR, f"sequences/{sequence}/image_2")))
frames = range(0, image_count, 1)
lengths = (int(image_count * 0.8), int(image_count * 0.1), int(image_count * 0.1))  # 80/10/10 split

# Fix seeds for reproducibility
seed = 1
torch.backends.cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Load the Rosario dataset using UnDeepVO's own DatasetManager (no pykitti)
dataset_manager = UnsupervisedDatasetManager(
    dataset_path=MAIN_DIR,
    sequence=sequence,
    lengths=lengths
)

# Paths for saving
checkpoint_dir = f"/content/drive/MyDrive/undeepvo_rosario_checkpoints_{sequence}"
log_dir = os.path.join(checkpoint_dir, "logs")
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model config
max_depth = 100
min_depth = 1
resnet = True
model = UnDeepVO(max_depth, min_depth, resnet=resnet).to(device)

# Training hyperparameters
num_epochs = 30
lr = 1e-4
batch_size = 4
betta1, betta2 = 0.9, 0.99
weight_decay = 0
lambda_position = 0.01
lambda_rotation = 0.0
lambda_s = 0.85
lambda_disparity = 0.2
lambda_registration = 1e-6
use_truth_poses = False

# Criterion
criterion = UnsupervisedCriterion(
    dataset_manager.get_cameras_calibration(device),
    lambda_position, lambda_rotation, lambda_s,
    lambda_disparity, lambda_registration
)

# Training handler
handler = TrainingProcessHandler(
    data_folder=log_dir,
    model_folder=checkpoint_dir,
    mlflow_tags={"name": "UnDeepVO_Training"},
    mlflow_parameters={
        "lr": lr,
        "lambda_position": lambda_position,
        "lambda_rotation": lambda_rotation,
        "lambda_s": lambda_s,
        "lambda_disparity": lambda_disparity,
        "lambda_registration": lambda_registration,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "betta2": betta2,
        "betta1": betta1,
        "max_depth": max_depth,
        "min_depth": min_depth,
        "use_truth_poses": use_truth_poses
    }
)

# Optimizer manager
scheduler_config = {"step_size": num_epochs // 5, "gamma": 0.5}
optimizer_manager = OptimizerManager(
    optimizer_class=torch.optim.Adam,
    scheduler_class=torch.optim.lr_scheduler.StepLR,
    scheduler_config=scheduler_config,
    lr=lr, betas=(betta1, betta2),
    weight_decay=weight_decay
)

# Define training problem
problem = UnsupervisedDepthProblem(
    model, criterion, optimizer_manager,
    dataset_manager, handler,
    batch_size=batch_size, name="undeepvo",
    use_truth_poses=use_truth_poses
)

# Run training
print(f"Starting training on Rosario sequence {sequence} with {image_count} images...")
problem.train(num_epochs)

# Save final model
final_checkpoint_path = os.path.join(checkpoint_dir, f"{handler._run_name}_final.pth")
torch.save(model.state_dict(), final_checkpoint_path)
print(f"Training completed. Final model saved at {final_checkpoint_path}")
