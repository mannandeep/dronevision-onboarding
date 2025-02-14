import os
import json
import cv2
import xml.etree.ElementTree as ET
from tqdm import tqdm

# Define paths
IMAGE_DIR = "..datasets/semantic_drone_dataset/trainingset/images/"
ANNOTATION_DIR = "..datasets/semantic_drone_dataset/trainingset/gt/boundingbox/labelme_xml/"
OUTPUT_JSON = "coco_annotations.json"

# COCO format placeholders
coco_dataset = {
    "info": {
        "description": "Semantic Drone Dataset in COCO Format",
        "version": "1.0",
        "year": 2025,
        "contributor": "Mannan",
        "date_created": "2025-02-14"
    },
    "licenses": [
        {"id": 1, "name": "CC0: Public Domain"}
    ],
    "images": [],
    "annotations": [],
    "categories": []
}

# Define class labels (from dataset documentation)
CLASS_NAMES = [
    "unlabeled", "paved-area", "dirt", "grass", "gravel", "water", "rocks",
    "pool", "vegetation", "roof", "wall", "window", "door", "fence",
    "fence-pole", "person", "dog", "car", "bicycle", "tree", "bald-tree",
    "ar-marker", "obstacle"
]

# Map class names to COCO categories
category_mapping = {name: i + 1 for i, name in enumerate(CLASS_NAMES)}

# Add categories to COCO format
for name, id in category_mapping.items():
    coco_dataset["categories"].append({"id": id, "name": name, "supercategory": "object"})

# Process images and annotations
annotation_id = 1  # Unique ID for each annotation
image_id = 1  # Unique ID for each image

for xml_file in tqdm(os.listdir(ANNOTATION_DIR), desc="Converting to COCO"):
    if not xml_file.endswith(".xml"):
        continue

    xml_path = os.path.join(ANNOTATION_DIR, xml_file)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Get image filename and size
    filename = root.find("filename").text
    img_path = os.path.join(IMAGE_DIR, filename)

    if not os.path.exists(img_path):
        continue  # Skip missing images

    img = cv2.imread(img_path)
    height, width, _ = img.shape

    # Add image entry
    coco_dataset["images"].append({
        "id": image_id,
        "file_name": filename,
        "width": width,
        "height": height
    })

    # Process each object in the XML file
    for obj in root.findall("object"):
        class_name = obj.find("name").text
        if class_name not in category_mapping:
            continue

        category_id = category_mapping[class_name]
        bbox_xml = obj.find("bndbox")

        xmin = int(float(bbox_xml.find("xmin").text))
        ymin = int(float(bbox_xml.find("ymin").text))
        xmax = int(float(bbox_xml.find("xmax").text))
        ymax = int(float(bbox_xml.find("ymax").text))

        width = xmax - xmin
        height = ymax - ymin

        # Add annotation entry
        coco_dataset["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [xmin, ymin, width, height],
            "area": width * height,
            "iscrowd": 0
        })

        annotation_id += 1

    image_id += 1

# Save to JSON file
with open(OUTPUT_JSON, "w") as f:
    json.dump(coco_dataset, f, indent=4)

print(f"Conversion complete! COCO dataset saved as {OUTPUT_JSON}")
