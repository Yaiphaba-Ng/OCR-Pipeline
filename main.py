import json, html
from pathlib import Path
from modules import detector, bbox_merger, recognizer

def process_image(image_path):
    """
    Processes a single image and returns the recognized text as a JSON string.

    Args:
        image_path (str): Path to the input image.

    Returns:
        str: JSON string containing the recognized text.
    """

    # 1. Detect words
    detector_model_file = "checkpoints/FAST/fast_base_20240926-134200_epoch100.pt"
    detection_output = detector.detect_words(detector_model_file, image_path)

    # 2. Merge bounding boxes
    merged_coco_output = bbox_merger.merge_annotations(detection_output)

    # 3. Recognize text
    recognizer_model_path = "checkpoints/TrOCR"
    recognizer_output = recognizer.recognize_text(merged_coco_output, image_path, recognizer_model_path)

    # Save to a file (optional)
    with open("recog_data.json", "w") as f:
        f.write(recognizer_output)

    json_file_path = Path("recog_data.json")

    # Load the JSON file
    with json_file_path.open() as f:
        data = json.load(f)

    # Escape HTML special characters
    escaped_output = [html.escape(item) for item in data]

    # Concatenate the strings with newlines
    output_string = "\n".join(escaped_output)

    # Create the JSON structure
    data = {"output": output_string}

    # Convert to JSON string
    json_data = json.dumps(data, indent=4)

    # Print the JSON data
    print(json_data)

    # Save to a file (optional)
    with open("recog_data.json", "w") as f:
        f.write(json_data)

    return recognizer_output

# (Optional) Example usage if you run main.py directly
if __name__ == "__main__":
    image_path = "files/input/0001_front.jpg"
    recognizer_output = process_image(image_path)

    # Save to a file (optional)
    with open("recog_data.json", "w") as f:
        f.write(recognizer_output)