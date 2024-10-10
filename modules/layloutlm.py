import json
import torch
from PIL import Image
import numpy as np
from transformers import AutoModelForTokenClassification, AutoProcessor

def load_layoutlm(layoutlm_model_path):
    # Load the model and processor for NER
    model = AutoModelForTokenClassification.from_pretrained(layoutlm_model_path)
    processor = AutoProcessor.from_pretrained(layoutlm_model_path)

    return model, processor

def assign_tag(coco_data, image_path, layoutlm_models):
    model = layoutlm_models[0]
    processor = layoutlm_models[1]
    # Load your image
    img = img = Image.open(image_path).convert("RGB")
    image_height = coco_data['images'][0]['height']  # Get height from JSON
    image_width = coco_data['images'][0]['width']  # Get width from JSON
    print("Original Image Size:", image_width, image_height)  # Debug print

    # Initialize lists for tokens and bounding boxes
    tokens = []
    boxes = []

    # Scaling factors to fit into the 0-1000 range
    scale_x = 1000 / image_width
    scale_y = 1000 / image_height

    # Extract tokens and bounding boxes from the annotations
    for annotation in coco_data['annotations']:
        # Append the 'trocr' value as the token
        tokens.append(annotation['trocr'])
        
        # Extract the bounding box
        bbox = annotation['bbox']
        x_min, y_min, width, height = bbox
        x_max = x_min + width
        y_max = y_min + height
        
        # Ensure valid bounding box
        if x_min < x_max and y_min < y_max:
            # Scale bounding box coordinates to 0-1000 range
            scaled_bbox = [
                x_min * scale_x,
                y_min * scale_y,
                x_max * scale_x,
                y_max * scale_y
            ]
            boxes.append(scaled_bbox)
        else:
            print(f"Invalid bounding box for token '{annotation['trocr']}': [{x_min}, {y_min}, {x_max}, {y_max}]")

    # Filter invalid bounding boxes (if needed)
    valid_tokens = []
    valid_boxes = []

    for token, box in zip(tokens, boxes):
        if box != [0, 0, 0, 0]:  # Ensure the bounding box is not empty
            valid_tokens.append(token)
            valid_boxes.append(box)

    # Convert valid boxes to tensor
    boxes_tensor = torch.tensor(valid_boxes, dtype=torch.float32)  # Use float for bounding box coordinates

    # for debugging
    # Print results
    # print("Valid Tokens:", valid_tokens)
    # print("Valid Bounding Boxes:", boxes_tensor)

    # def unnormalize_box(bbox, width, height):
    #     return [
    #         width * (bbox[0] / 1000),
    #         height * (bbox[1] / 1000),
    #         width * (bbox[2] / 1000),
    #         height * (bbox[3] / 1000),
    #     ]

    if len(valid_tokens) != len(boxes_tensor):
        print(f"Warning: Number of tokens ({len(valid_tokens)}) does not match number of boxes ({len(boxes_tensor)})!")
    else:
        # Convert boxes_tensor to long type before passing to the processor
        boxes_tensor = boxes_tensor.long()  # Change to long type

        # Create encoding for the model without padding
        encoding = processor(
            img, 
            valid_tokens, 
            boxes=boxes_tensor, 
            return_tensors="pt", 
            padding=False  # Disable automatic padding
        )

        # Check the shape of input tensors after encoding
        # print(f"input_ids shape: {encoding['input_ids'].shape}")
        # print(f"attention_mask shape: {encoding['attention_mask'].shape}")
        # print(f"bbox shape: {encoding['bbox'].shape}")
        # print(f"pixel_values shape: {encoding['pixel_values'].shape}")

        # Run inference
        with torch.no_grad():
            outputs = model(**encoding)

        logits = outputs.logits
        predictions = logits.argmax(-1).squeeze().tolist()

        # Print some predictions for debugging
        #print(f"Predictions: {predictions[:5]}")

        encoded_valid_tokens = encoding['input_ids'].squeeze().tolist()
        encoded_valid_boxes = encoding['bbox'].squeeze().tolist()

        # Initialize an empty dictionary to store the final results
        prediction_dict = {}

        # Remove the first and last indices, and iterate over the remaining ones
        for idx, (token, box, pred) in enumerate(zip(encoded_valid_tokens[1:-1], encoded_valid_boxes[1:-1], predictions[1:-1])):
            # Decode the token
            decoded_token = processor.tokenizer.decode([token], skip_special_tokens=True)

            # Get the prediction label from the model's config
            label_name = model.config.id2label[pred]

            # Append or create the token list for the prediction key in the dictionary
            if label_name in prediction_dict:
                prediction_dict[label_name] += f"{decoded_token}"  # Concatenate tokens for the same prediction
            else:
                prediction_dict[label_name] = decoded_token  # Create new entry for the prediction

        # Clean up spaces and punctuation
        for label in prediction_dict:
            # Remove spaces, dots, and any unwanted characters
            cleaned_tokens = prediction_dict[label].replace('.', '').replace(' ', '')  # Adjust as necessary
            prediction_dict[label] = cleaned_tokens



    return prediction_dict

if __name__ == "__main__":
    image_path = "files/input/1.jpg"
    json_file_path = "files/input/trocr_coco.json"
    # Load the JSON data
    with open(json_file_path, 'r') as file:
        coco_data = json.load(file)
    # print(json.dumps(coco_data, indent=4))
    layoutlm_model_path = "checkpoints/LayoutLM"
    llm_model, llm_processor = load_layoutlm(layoutlm_model_path)
    layoutlm_models = [llm_model, llm_processor]
    prediction_dict = assign_tag(coco_data, image_path, layoutlm_models)
    # Print the cleaned dictionary
    for label, tokens in prediction_dict.items():
        print(f"{label}: {tokens}")
    # Store prediction_dict to a JSON file
    with open('files/input/layoutlm_output.json', 'w') as json_file:
        json.dump(prediction_dict, json_file, indent=4)