import os

sequence = '01'
base_path = f'rosario_dataset/sequences/{sequence}'

# Define folders
image_2_folder = os.path.join(base_path, 'image_2')
image_3_folder = os.path.join(base_path, 'image_3')

# Count .png images only
left_images = [f for f in os.listdir(image_2_folder) if f.endswith('.png')]
right_images = [f for f in os.listdir(image_3_folder) if f.endswith('.png')]

print(f"Sequence {sequence} - image_2 (left):  {len(left_images)} images")
print(f"Sequence {sequence} - image_3 (right): {len(right_images)} images")
