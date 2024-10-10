import json
from typing import List, Dict, Tuple

# def load_coco_annotations(file_path: str) -> Dict:
#     with open(file_path, "r") as f:
#         return json.load(f)

def box_overlap(box1: List[float], box2: List[float]) -> bool:
    left1, top1, width1, height1 = box1
    left2, top2, width2, height2 = box2
    return not (
        left1 + width1 < left2
        or left2 + width2 < left1
        or top1 + height1 < top2
        or top2 + height2 < top1
    )

def merge_boxes(boxes: List[List[float]]) -> List[float]:
    left = min(box[0] for box in boxes)
    top = min(box[1] for box in boxes)
    right = max(box[0] + box[2] for box in boxes)
    bottom = max(box[1] + box[3] for box in boxes)
    return [left, top, right - left, bottom - top]

def process_image_annotations(
    annotations: List[Dict], vertical_threshold: float, overlap_threshold: float
) -> List[Dict]:
    # Sort annotations by y-coordinate (top to bottom)
    sorted_annotations = sorted(annotations, key=lambda x: x["bbox"][1])

    lines = []
    current_line = []

    for ann in sorted_annotations:
        if (
            not current_line
            or abs(ann["bbox"][1] - current_line[-1]["bbox"][1]) <= vertical_threshold
        ):
            current_line.append(ann)
        else:
            lines.append(current_line)
            current_line = [ann]

    if current_line:
        lines.append(current_line)

    merged_annotations = []

    for line in lines:
        # Sort annotations in the line by x-coordinate (left to right)
        line.sort(key=lambda x: x["bbox"][0])

        merged_line = []
        current_group = []

        for i, ann in enumerate(line):
            if not current_group:
                current_group.append(ann)
            else:
                last_box = current_group[-1]["bbox"]
                current_box = ann["bbox"]

                # Check if the current box overlaps horizontally with the last box
                if box_overlap(last_box, current_box):
                    current_group.append(ann)
                else:
                    # Check if there's a box to the right that intersects with an imaginary line
                    center_y = last_box[1] + last_box[3] / 2
                    found_intersection = False
                    for next_ann in line[i:]:
                        next_box = next_ann["bbox"]
                        if (
                            next_box[1] <= center_y <= next_box[1] + next_box[3]
                            and next_box[0] > last_box[0] + last_box[2]
                        ):
                            current_group.extend(line[i:])
                            found_intersection = True
                            break

                    if found_intersection:
                        break
                    else:
                        merged_line.append(
                            merge_boxes([box["bbox"] for box in current_group])
                        )
                        current_group = [ann]

        if current_group:
            merged_line.append(merge_boxes([box["bbox"] for box in current_group]))

        # Create new annotations for the merged boxes
        for merged_box in merged_line:
            new_ann = line[0].copy()  # Use the first annotation as a template
            new_ann["bbox"] = merged_box
            new_ann["area"] = merged_box[2] * merged_box[3]
            merged_annotations.append(new_ann)

    # Additional pass to merge overlapping boxes
    final_annotations = []
    processed = set()

    for i, ann in enumerate(merged_annotations):
        if i in processed:
            continue

        current_group = [ann]
        processed.add(i)

        for j, other_ann in enumerate(merged_annotations):
            if i == j or j in processed:
                continue

            if box_overlap(ann["bbox"], other_ann["bbox"]):
                overlap_area = calculate_overlap_area(ann["bbox"], other_ann["bbox"])
                min_area = min(ann["area"], other_ann["area"])
                if overlap_area / min_area > overlap_threshold:
                    current_group.append(other_ann)
                    processed.add(j)

        if len(current_group) > 1:
            merged_box = merge_boxes([box["bbox"] for box in current_group])
            new_ann = max(current_group, key=lambda x: x["area"]).copy()
            new_ann["bbox"] = merged_box
            new_ann["area"] = merged_box[2] * merged_box[3]
            final_annotations.append(new_ann)
        else:
            final_annotations.append(current_group[0])

    return final_annotations

def calculate_overlap_area(box1: List[float], box2: List[float]) -> float:
    left1, top1, width1, height1 = box1
    left2, top2, width2, height2 = box2

    x_overlap = max(0, min(left1 + width1, left2 + width2) - max(left1, left2))
    y_overlap = max(0, min(top1 + height1, top2 + height2) - max(top1, top2))

    return x_overlap * y_overlap

def merge_annotations(
    coco_data,
    # output_file_path: str,
    vertical_threshold: float = 4,
    overlap_threshold: float = 0.5,
):
    # Load COCO annotations
    # coco_data = load_coco_annotations(coco_file_path)

    # Group and merge annotations for each image
    new_annotations = []
    for image in coco_data["images"]:
        image_id = image["id"]
        image_annotations = [
            ann for ann in coco_data["annotations"] if ann["image_id"] == image_id
        ]

        merged_annotations = process_image_annotations(
            image_annotations, vertical_threshold, overlap_threshold
        )
        new_annotations.extend(merged_annotations)

    # Replace original annotations with merged ones
    coco_data["annotations"] = new_annotations

    return new_annotations

if __name__ == "__merge_annotations__":
    coco_file_path = "files/input/doctr_coco.json"
    coco_data = merge_annotations(coco_file_path)
    # Save the updated COCO annotations
    with open("files/input/box_merged_coco.json", "w") as f:
        json.dump(coco_data, f, indent=2)