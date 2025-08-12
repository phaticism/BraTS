import os
import shutil
import logging
import json
import random

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

src_dir = "2D/data/BraTS2021o"
dst_dir = "./2D/data/BraTS2021"

# Create partitions
partitions = ["001", "002", "003"]
for partition in partitions:
    partition_dir = os.path.join(dst_dir, partition)
    os.makedirs(os.path.join(partition_dir, "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(partition_dir, "labelsTr"), exist_ok=True)

successful_dirs = 0
training_data = []

# Get all subdirectories and shuffle them
subdirs = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]
random.shuffle(subdirs)

# Calculate items per partition
items_per_partition = len(subdirs) // 3
partition_ranges = [
    (0, items_per_partition),
    (items_per_partition, 2 * items_per_partition),
    (2 * items_per_partition, len(subdirs))
]

# Dictionary to store training data for each partition
partition_training_data = {p: [] for p in partitions}

for i, subdir in enumerate(subdirs):
    # Determine which partition this item belongs to
    partition_idx = min(i // items_per_partition, 2)
    partition = partitions[partition_idx]
    partition_dir = os.path.join(dst_dir, partition)
    
    subdir_path = os.path.join(src_dir, subdir)
    logging.info(f"Processing directory: {subdir} for partition {partition}")
    success = True

    # Copy the _flair.nii.gz file to imagesTr
    flair_file = os.path.join(subdir_path, f"{subdir}_flair.nii.gz")
    if os.path.exists(flair_file):
        new_flair_file = os.path.join(partition_dir, "imagesTr", f"{subdir}.nii.gz")
        shutil.copy(flair_file, new_flair_file)
        logging.info(f"Copied {flair_file} to {new_flair_file}")
    else:
        logging.warning(f"{flair_file} does not exist")
        success = False

    # Copy the _seg.nii.gz file to labelsTr
    seg_file = os.path.join(subdir_path, f"{subdir}_seg.nii.gz")
    if os.path.exists(seg_file):
        new_seg_file = os.path.join(partition_dir, "labelsTr", f"{subdir}.nii.gz")
        shutil.copy(seg_file, new_seg_file)
        logging.info(f"Copied {seg_file} to {new_seg_file}")
    else:
        logging.warning(f"{seg_file} does not exist")
        success = False

    if success:
        # Add to global training data
        training_data.append({
            "image": os.path.join(partition, "imagesTr", f"{subdir}.nii.gz"),
            "label": os.path.join(partition, "labelsTr", f"{subdir}.nii.gz")
        })
        # Add to partition-specific training data
        partition_training_data[partition].append({
            "image": os.path.join("imagesTr", f"{subdir}.nii.gz"),
            "label": os.path.join("labelsTr", f"{subdir}.nii.gz")
        })
        successful_dirs += 1

# Create dataset.json for each partition
for partition in partitions:
    partition_dir = os.path.join(dst_dir, partition)
    partition_dataset_info = {
        "name": f"BRATS2021_Partition_{partition}",
        "description": "Gliomas segmentation tumour and oedema in on brain images",
        "reference": "http://braintumorsegmentation.org/",
        "licence": "CC-BY-SA 4.0",
        "release": "Not provided",
        "tensorImageSize": "3D",
        "modality": {
            "0": "FLAIR",
            "1": "T1w",
            "2": "t1gd",
            "3": "T2w"
        },
        "labels": {
            "0": "background",
            "1": "non-enhancing tumor",
            "2": "edema",
            "4": "enhancing tumour"
        },
        "numTraining": len(partition_training_data[partition]),
        "numTest": "Not provided",
        "training": partition_training_data[partition],
        "test": "Not provided"
    }

    with open(os.path.join(partition_dir, "dataset.json"), "w") as f:
        json.dump(partition_dataset_info, f, indent=4)
    logging.info(f"Created dataset.json for partition {partition}")

# Create main dataset.json
dataset_info = {
    "name": "BRATS2021",
    "description": "Gliomas segmentation tumour and oedema in on brain images",
    "reference": "http://braintumorsegmentation.org/",
    "licence": "CC-BY-SA 4.0",
    "release": "Not provided",
    "tensorImageSize": "3D",
    "modality": {
        "0": "FLAIR",
        "1": "T1w",
        "2": "t1gd",
        "3": "T2w"
    },
    "labels": {
        "0": "background",
        "1": "non-enhancing tumor",
        "2": "edema",
        "4": "enhancing tumour"
    },
    "numTraining": successful_dirs,
    "numTest": "Not provided",
    "training": training_data,
    "test": "Not provided"
}

with open(os.path.join(dst_dir, "dataset.json"), "w") as f:
    json.dump(dataset_info, f, indent=4)

logging.info(
    f"Files copied successfully. Total successful directories: {successful_dirs}"
)
logging.info("dataset.json files created successfully.")
