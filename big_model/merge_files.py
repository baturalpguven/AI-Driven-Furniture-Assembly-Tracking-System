import os
import shutil

def merge_directories(src, dst):
    """ Recursively merge directories from src to dst """
    for item in os.listdir(src):
        src_path = os.path.join(src, item)
        dst_path = os.path.join(dst, item)

        if os.path.isdir(src_path):
            # Create the directory in the destination if it does not exist
            os.makedirs(dst_path, exist_ok=True)
            # Recursively merge the contents
            merge_directories(src_path, dst_path)
        else:
            # Check if the file exists in destination
            if not os.path.exists(dst_path):
                # Copy the file to the destination directory
                shutil.copy2(src_path, dst_path)
            else:
                print(f"Skipping existing file: {dst_path}")

# Base directory where all folders are located
base_dir = '/root/maaf_umer/Data_Segmented_Tracked_Action_JPG'

# Source directory containing subdirectories to be merged
source_dir = os.path.join(base_dir, 'Data_Segmented_Tracked_Action')

# Iterate over each action category directory in the source directory
for action_category in os.listdir(source_dir):
    src_folder = os.path.join(source_dir, action_category)
    dst_folder = os.path.join(base_dir, action_category)

    # Ensure the destination folder exists
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    # Merge the directories
    merge_directories(src_folder, dst_folder)

print("Directories have been successfully merged.")
