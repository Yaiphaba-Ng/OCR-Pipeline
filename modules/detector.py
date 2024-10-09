import os
import torch
import json
from PIL import Image
from doctr.io import DocumentFile
from doctr.models import ocr_predictor, fast_base, detection_predictor

os.environ["USE_TORCH"] = "1"
print(os.environ["USE_TORCH"])

def detect_words(detector_model_file, image_path):
    """
    Detects words in an image and returns the COCO JSON output as a variable.

    Args:
        image_path (str): Path to the input image file.

    Returns:
        dict: COCO JSON output containing word bounding boxes.
    """

    # Initialize the default OCR predictor
    # default_predictor = detection_predictor('fast_base',pretrained=True)

    # Load custom detection model
    # detector_model_file = "checkpoints/FAST/fast_base_20240926-134200_epoch100.pt"
    det_model = fast_base(pretrained=False, pretrained_backbone=False)
    det_params = torch.load(detector_model_file, map_location="cpu")
    det_model.load_state_dict(det_params)

    trained_predictor = ocr_predictor(
        det_arch=det_model, 
        reco_arch="crnn_vgg16_bn", 
        pretrained=True
    )
    # model = default_predictor
    model = trained_predictor

    # Initialize COCO data structure
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": [{
            "id": 1,
            "name": "word",
            "supercategory": "text",
        }]
    }

    annotation_id = 1  # Unique annotation ID
    image_id = 1  # Unique image ID

    # Load the document and image
    doc = DocumentFile.from_images(image_path)
    image = Image.open(image_path)
    width, height = image.size

    # Add image info to COCO structure
    coco_output["images"].append(
        {
            "file_name": os.path.basename(image_path),
            "height": height,
            "width": width,
            "id": image_id,
        }
    )

    # Get the result from the model
    result = model(doc)
    # Access the blocks of text in the document prediction
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    # Get the bounding box
                    bbox = word.geometry  # [x_min, y_min, x_max, y_max]
                    bbox_coco = [
                        bbox[0][0] * width,  # x_min
                        bbox[0][1] * height,  # y_min
                        (bbox[1][0] - bbox[0][0]) * width,  # width
                        (bbox[1][1] - bbox[0][1]) * height,  # height
                    ]

                    # Append annotation to COCO structure
                    coco_output["annotations"].append(
                        {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": 1,  # Word category
                            "bbox": bbox_coco,
                            "area": bbox_coco[2] * bbox_coco[3],
                            "iscrowd": 0,
                        }
                    )

                    annotation_id += 1

    return coco_output

# Example usage within the script (optional):
if __name__ == "__main__":
    image_path = "files/input/1.jpg"
    output_json = detect_words(image_path)
    print(json.dumps(output_json, indent=4))  # Print the JSON output