import json
import os
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as v2
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
# import matplotlib.pyplot as plt
# from typing import List, Dict, Tuple
# from dataclasses import dataclass

def load_coco_annotations(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        return json.load(f)

def preprocess_image(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8)
    denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
    rgb = cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb)

def load_model_and_processor(model_path):
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    return processor, model, device

def process_image(img, bbox, processor, model, device, transform):
    # Crop image based on bounding box
    x, y, w, h = map(int, bbox)
    cropped_img = img.crop((x, y, x+w, y+h))

    # Preprocess
    preprocessed_img = preprocess_image(cropped_img)

    # Transform and move to device
    img_t = transform(preprocessed_img).unsqueeze(0).to(device)

    # Generate text
    with torch.no_grad():
        generated_ids = model.generate(img_t)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_text

def main(coco_file_path, image_folder, model_path):
    # Load COCO annotations
    coco_data = load_coco_annotations(coco_file_path)

    # Load model and processor
    processor, model, device = load_model_and_processor(model_path)

    # Define transform
    transform = v2.Compose([
        v2.Resize((384,384)),
        v2.ToTensor(),
        v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # Process each image
    for image_info in coco_data['images']:
        image_path = os.path.join(image_folder, image_info['file_name'])
        img = Image.open(image_path).convert("RGB")

        # Get annotations for this image
        image_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_info['id']]

        # Sort annotations from top-left to bottom-right
        image_annotations.sort(key=lambda x: (x['bbox'][1], x['bbox'][0]))

        print(f"Processing image: {image_info['file_name']}")

        # # Display the full image
        # plt.figure(figsize=(12, 12))
        # plt.imshow(img)
        # plt.axis('off')
        # plt.title(f"Full Image: {image_info['file_name']}")
        # plt.show()

        # Process each bounding box and print recognized text
        for ann in image_annotations:
            generated_text = process_image(img, ann['bbox'], processor, model, device, transform)
            print(f"Recognized Text: {generated_text}")

        print("\n")

if __name__ == "__main__":
    coco_file_path = "./utils/notebooks/data/merged_coco_result.json"
    image_folder = "./files/input/Single Sample Inference"
    model_path = "./checkpoints/TrOCR"
    main(coco_file_path, image_folder, model_path)