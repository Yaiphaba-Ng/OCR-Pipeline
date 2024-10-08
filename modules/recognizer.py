import cv2, torch, json, copy
import numpy as np
from PIL import Image
import torchvision.transforms as v2
from transformers import VisionEncoderDecoderModel, TrOCRProcessor

def preprocess_image(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8
    )
    denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
    rgb = cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb)

def load_recognizer(model_path):
    processor = TrOCRProcessor.from_pretrained(
        "microsoft/trocr-large-handwritten"
    )
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
    generated_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0]

    return generated_text

def recognize_text(coco_data, image_path, recogniser_models):
    processor = recogniser_models[0]
    model = recogniser_models[1]
    device = recogniser_models[2]

    transform = v2.Compose([
        v2.Resize((384,384)),
        v2.ToTensor(),
        v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    img = Image.open(image_path).convert("RGB")

    # Get annotations for this image
    image_annotations = coco_data['annotations']

    # Sort annotations from top-left to bottom-right
    image_annotations.sort(key=lambda x: (x['bbox'][1], x['bbox'][0]))

    # Process each bounding box and store recognized text
    results = []
    updated_coco_data = copy.deepcopy(coco_data)

    for ann in updated_coco_data['annotations']:
        generated_text = process_image(
            img, ann['bbox'], processor, model, device, transform
        )
        results.append(generated_text)
        ann['trocr'] = generated_text

    return results, updated_coco_data

# Example usage within the script (optional):
if __name__ == "__main__":
    coco_file_path = "files/input/box_merged_coco.json"
    image_path = "files/input/1.jpg"
    model_path = "checkpoints/TrOCR"
    
    # Load COCO data from file (for testing purposes)
    with open(coco_file_path, 'r') as f:
        coco_data = json.load(f)
    # Load model and processor
    processor, model, device = load_recognizer(model_path)
    recognizer_models = [processor, model, device]
    trocr_result, trocr_coco = recognize_text(coco_data, image_path, recognizer_models)
    print(trocr_result)

    with open("files/input/trocr_result.json", "w") as file:
        json.dump(trocr_result, file)
    with open("files/input/trocr_coco.json", "w") as file:
        json.dump(trocr_coco, file)