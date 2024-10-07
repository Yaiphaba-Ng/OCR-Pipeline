from typing import List, Dict
from dataclasses import dataclass
import json

@dataclass
class BoundingBox:
    x: float
    y: float
    width: float
    height: float

    @property
    def left(self) -> float:
        return self.x

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def top(self) -> float:
        return self.y

    @property
    def bottom(self) -> float:
        return self.y + self.height

def merge_line_boxes(
    boxes: List[BoundingBox], distance_threshold: float = 20
) -> List[BoundingBox]:
    if not boxes:
        return []

    # Sort boxes by their top-left corner
    sorted_boxes = sorted(boxes, key=lambda b: (b.top, b.left))
    merged_boxes = []
    current_line = [sorted_boxes[0]]

    for box in sorted_boxes[1:]:
        last_box = current_line[-1]

        # Check if the box is on the same line
        if (
            abs(box.top - last_box.top) <= distance_threshold
            and abs(box.bottom - last_box.bottom) <= distance_threshold
            and (box.left - last_box.right) <= distance_threshold
        ):
            current_line.append(box)
        else:
            # Merge the current line and start a new one
            merged_boxes.append(merge_boxes(current_line))
            current_line = [box]

    # Merge the last line
    if current_line:
        merged_boxes.append(merge_boxes(current_line))

    return merged_boxes

def merge_boxes(boxes: List[BoundingBox]) -> BoundingBox:
    left = min(box.left for box in boxes)
    top = min(box.top for box in boxes)
    right = max(box.right for box in boxes)
    bottom = max(box.bottom for box in boxes)
    return BoundingBox(left, top, right - left, bottom - top)

def process_coco_annotations(annotations: List[Dict]) -> List[BoundingBox]:
    boxes = [BoundingBox(*ann["bbox"]) for ann in annotations]
    return merge_line_boxes(boxes)

def create_merged_coco_annotations(
    original_coco: Dict, merged_boxes: List[BoundingBox]
) -> Dict:
    merged_coco = original_coco.copy()
    merged_coco["annotations"] = [
        {
            "id": i + 1,
            "image_id": original_coco["annotations"][0]["image_id"],
            "category_id": 1,
            "bbox": [box.x, box.y, box.width, box.height],
            "area": box.width * box.height,
            "iscrowd": 0,
        }
        for i, box in enumerate(merged_boxes)
    ]
    merged_coco["categories"] = [
        {"id": 1, "name": "text_line", "supercategory": "text"}
    ]
    return merged_coco

def merge_annotations(input_coco_data: Dict) -> Dict:
    """
    Merges bounding boxes in COCO annotations and returns the merged COCO JSON.

    Args:
        input_coco_data (dict): COCO annotations as a dictionary.

    Returns:
        dict: Merged COCO JSON output.
    """
    merged_boxes = process_coco_annotations(input_coco_data["annotations"])

    print(f"Number of original boxes: {len(input_coco_data['annotations'])}")
    print(f"Number of merged line boxes: {len(merged_boxes)}")

    merged_coco = create_merged_coco_annotations(input_coco_data, merged_boxes)
    
    return merged_coco

# Example usage within the script (optional):
if __name__ == "__main__":
    # Load COCO data from a file for testing (simulate input from detector)
    with open("./files/input/Single Sample Inference/detection.json", "r") as f:
        input_coco_data = json.load(f)

    merged_coco = merge_annotations(input_coco_data)
    print(merged_coco)