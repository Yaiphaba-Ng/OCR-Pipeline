from doctr.models import ocr_predictor
from PIL import Image
import torch
from doctr.models import ocr_predictor, fast_base
import numpy as np
from transformers import AutoModelForTokenClassification, AutoProcessor
# from transformers import LayoutLMv3Processor
# import torch.nn.functional as F

det_model = fast_base(pretrained=False, pretrained_backbone=False)
det_params = torch.load('F:/LayoutLMV3/pretrained_model/fast_base_20240926-134200_epoch100.pt', map_location="cpu")
det_model.load_state_dict(det_params)
ocr_model = ocr_predictor(det_arch=det_model, reco_arch="crnn_vgg16_bn", pretrained=True)

# Load your image
img = Image.open('layoutlmv3/0001_front.jpg')
width, height = img.size
print("Original Image Size:", width, height)  # Debug print
img_np = np.array(img)

# Perform OCR on the image
result = ocr_model([img_np])

# Check the overall structure of the OCR result
# print("OCR Result Structure:", result)

# Extract tokens and bounding boxes
tokens = []
boxes = []

for block in result.pages[0].blocks:
    for line in block.lines:
        for word in line.words:
            tokens.append(word.value)
            geometry = word.geometry
            
            if geometry:
                # Calculate bounding box coordinates
                x_min = min(p[0] for p in geometry)
                y_min = min(p[1] for p in geometry)
                x_max = max(p[0] for p in geometry)
                y_max = max(p[1] for p in geometry)

                # Check for valid bounding box
                if x_min < x_max and y_min < y_max:
                    # Scale to 0-1000 range
                    boxes.append([x_min * 1000, y_min * 1000, x_max * 1000, y_max * 1000])
                else:
                    print(f"Invalid bounding box for word '{word.value}': [{x_min}, {y_min}, {x_max}, {y_max}]")
            else:
                print(f"No geometry found for word '{word.value}'")
                
# Filter invalid bounding boxes
valid_tokens = []
valid_boxes = []

for token, box in zip(tokens, boxes):
    if box != [0, 0, 0, 0]:
        valid_tokens.append(token)
        valid_boxes.append(box)

boxes_tensor = torch.tensor(valid_boxes, dtype=torch.int64)

# Load the model and processor for NER
model = AutoModelForTokenClassification.from_pretrained("test/checkpoint-500")
processor = AutoProcessor.from_pretrained("test/checkpoint-500")

def unnormalize_box(bbox, width, height):
     return [
         width * (bbox[0] / 1000),
         height * (bbox[1] / 1000),
         width * (bbox[2] / 1000),
         height * (bbox[3] / 1000),
     ]
    
if len(valid_tokens) != len(boxes_tensor):
    print(f"Warning: Number of tokens ({len(valid_tokens)}) does not match number of boxes ({len(boxes_tensor)})!")
else:
    # Create encoding for the model without padding
    encoding = processor(
        img, 
        valid_tokens, 
        boxes=boxes_tensor, 
        return_tensors="pt", 
        padding=False  # Disable automatic padding
    )

    # Check the shape of input tensors after encoding
#     print(f"input_ids shape: {encoding['input_ids'].shape}")
#     print(f"attention_mask shape: {encoding['attention_mask'].shape}")
#     print(f"bbox shape: {encoding['bbox'].shape}")
#     print(f"pixel_values shape: {encoding['pixel_values'].shape}")

    # Run inference
    with torch.no_grad():
        outputs = model(**encoding)

    logits = outputs.logits
    predictions = logits.argmax(-1).squeeze().tolist()

    # Print some predictions for debugging
#     print(f"Predictions: {predictions[:5]}")
encoded_valid_tokens = encoding['input_ids'].squeeze().tolist()
encoded_valid_boxes = encoding['bbox'].squeeze().tolist()
