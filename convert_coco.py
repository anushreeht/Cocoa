import json
import shutil
import random
from pathlib import Path

# ============================================================
# 1. ORIGINAL DATASET
# ============================================================

SOURCE = Path(
    r"C:\Users\ANUSHREE H T\Downloads\RipSetCocoaCNCH12\RipSetCocoaCNCH12\COCO_10"
)

# ============================================================
# 2. PROJECT LOCATION
# ============================================================

PROJECT = Path(r"D:\projectrcn")

OUTPUT = PROJECT / "dataset"

# ============================================================
# 3. CLASSES
# ============================================================

CLASSES = {
    1: 0,
    2: 1,
    3: 2,
    4: 3
}

# ============================================================
# 4. CREATE FOLDERS
# ============================================================

for split in ["train", "val", "test"]:

    (OUTPUT / "images" / split).mkdir(
        parents=True,
        exist_ok=True
    )

    (OUTPUT / "labels" / split).mkdir(
        parents=True,
        exist_ok=True
    )

# ============================================================
# 5. FIND UNIQUE IMAGES
# ============================================================

image_files = {}
    
for folder_name in [
    "images_C1",
    "images_C2",
    "images_C3",
    "images_C4"
]:

    folder = SOURCE / folder_name

    if not folder.exists():
        print("WARNING: Folder not found:", folder)
        continue

    for extension in [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.JPG",
        "*.JPEG",
        "*.PNG"
    ]:

        for image_path in folder.glob(extension):

            # Keep only one copy of each image name
            image_files[image_path.name] = image_path


image_files = list(image_files.values())

print("Total UNIQUE images found:", len(image_files))

# ============================================================
# 6. TRAIN / VAL / TEST SPLIT
# ============================================================

random.seed(42)

random.shuffle(image_files)

total = len(image_files)

train_end = int(total * 0.70)

val_end = int(total * 0.85)

train_images = image_files[:train_end]

val_images = image_files[train_end:val_end]

test_images = image_files[val_end:]

splits = {
    "train": train_images,
    "val": val_images,
    "test": test_images
}

print("Train:", len(train_images))
print("Validation:", len(val_images))
print("Test:", len(test_images))

# ============================================================
# 7. COPY IMAGES
# ============================================================

for split, files in splits.items():

    for image_path in files:

        destination = (
            OUTPUT
            / "images"
            / split
            / image_path.name
        )

        if not destination.exists():

            shutil.copy2(
                image_path,
                destination
            )

# ============================================================
# 8. COCO ANNOTATION FILES
# ============================================================

annotation_folder = SOURCE / "annotations"

annotation_files = [
    annotation_folder / "instances_default_C1.json",
    annotation_folder / "instances_default_C2.json",
    annotation_folder / "instances_default_C3.json",
    annotation_folder / "instances_default_C4.json"
]

# ============================================================
# 9. READ COCO ANNOTATIONS
# ============================================================

all_annotations = {}

for json_file in annotation_files:

    if not json_file.exists():

        print(
            "WARNING: annotation file not found:",
            json_file
        )

        continue

    print(
        "Reading:",
        json_file.name
    )

    with open(
        json_file,
        "r",
        encoding="utf-8"
    ) as file:

        coco = json.load(file)

    images_info = {
        image["id"]: image
        for image in coco.get("images", [])
    }

    for annotation in coco.get(
        "annotations",
        []
    ):

        image_id = annotation.get("image_id")

        if image_id not in images_info:
            continue

        image_info = images_info[image_id]

        image_name = image_info["file_name"]

        bbox = annotation.get("bbox")

        if not bbox:
            continue

        x, y, width, height = bbox

        image_width = image_info["width"]

        image_height = image_info["height"]

        # COCO → YOLO

        x_center = (
            x + width / 2
        ) / image_width

        y_center = (
            y + height / 2
        ) / image_height

        normalized_width = (
            width / image_width
        )

        normalized_height = (
            height / image_height
        )

        class_id = CLASSES.get(annotation.get("category_id"))

        if class_id is None:
            continue

        label = (
            f"{class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{normalized_width:.6f} "
            f"{normalized_height:.6f}"
        )

        if image_name not in all_annotations:
            all_annotations[image_name] = set()

        all_annotations[image_name].add(label)

# ============================================================
# 10. CREATE YOLO LABEL FILES
# ============================================================

for split, files in splits.items():

    for image_path in files:

        image_name = image_path.name

        labels = all_annotations.get(
            image_name,
            []
        )

        label_file = (
            OUTPUT
            / "labels"
            / split
            / f"{image_path.stem}.txt"
        )

        with open(
            label_file,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(
                "\n".join(sorted(labels))
            )

# ============================================================
# 11. FINISHED
# ============================================================

print()
print("==============================")
print("COCO → YOLO CONVERSION DONE")
print("==============================")
print()

print("Dataset created at:")
print(OUTPUT)

print()

print("Classes:")
print("0 = Unripe")
print("1 = Semiripe")
print("2 = Ripe")
print("3 = Overripe")