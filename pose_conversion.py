import numpy as np
from scipy.spatial.transform import Rotation as R

input_file = "Rosario_dataset_raw/sequence01_gt.txt"
output_file = "rosario_dataset/poses/01.txt"

poses = []

with open(input_file, "r") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) != 8:
            continue  # Skip malformed lines

        # Parse fields
        _, x, y, z, qx, qy, qz, qw = map(float, parts)

        # Convert quaternion to rotation matrix
        rot = R.from_quat([qx, qy, qz, qw])
        rot_matrix = rot.as_matrix()  # 3x3

        # Combine rotation + translation into 3x4 matrix
        pose_matrix = np.hstack((rot_matrix, np.array([[x], [y], [z]])))  # 3x4
        poses.append(pose_matrix)

# Optionally normalize (start at 0,0,0)
origin = poses[0][:, 3]
poses = [np.hstack((pose[:, :3], pose[:, 3:] - origin.reshape(3, 1))) for pose in poses]

# Save in KITTI format (one line per pose)
with open(output_file, "w") as f:
    for pose in poses:
        flattened = pose.flatten()
        f.write(" ".join(f"{v:.9e}" for v in flattened) + "\n")

print("✅ KITTI-style poses.txt written to:", output_file)
