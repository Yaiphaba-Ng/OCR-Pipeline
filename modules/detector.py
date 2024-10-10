import os
import torch
import json
from PIL import Image
from doctr.io import DocumentFile
from doctr.models import ocr_predictor, fast_base, detection_predictor

os.environ["USE_TORCH"] = "1"
print(os.environ["USE_TORCH"])

detector_model_file = "checkpoints/FAST/fast_base_20240926-134200_epoch100.pt"

def load_detector(detector_model_file):
    
    # Initialize the default OCR predictor
    default_predictor = detection_predictor('fast_base',pretrained=True)

    # Load custom detection model
    det_model = fast_base(pretrained=False, pretrained_backbone=False)
    det_params = torch.load(detector_model_file, map_location="cpu")
    det_model.load_state_dict(det_params)

    trained_predictor = ocr_predictor(
        det_arch=det_model, 
        reco_arch="crnn_vgg16_bn", 
        pretrained=True
    )

    # trained_predictor = default_predictor
    return trained_predictor, default_predictor

def detect_words(image_path, detector_models, model="trained"):

    trained_detector = detector_models[0]
    default_detector = detector_models[1]
    if model == "trained":
        selected_model = trained_detector
    elif model == "default":
        selected_model = default_detector
    else:
        raise ValueError("Invalid model selection. Choose 'trained' or 'default'.")


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
    result = selected_model(doc)
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
                            "confidence": word.confidence,  # Add confidence score
                            "doctr": word.value,  # Add word value

                        }
                    )

                    annotation_id += 1
    return coco_output, result

# Example usage within the script (optional):
if __name__ == "__main__":
    image_path = "files/input/1.jpg"
    trained_detector, default_detector = load_detector(detector_model_file)
    detector_models = [trained_detector, default_detector]
    doctr_coco, doctr_result = detect_words(image_path, detector_models, model="trained")
    # print(json.dumps(doctr_coco, indent=4))  # Print the JSON output
    with open('files/input/doctr_result.txt', 'w') as f:
        f.write(str(doctr_result))
    with open("files/input/doctr_coco.json", "w") as file:
        json.dump(doctr_coco, file)
