from modules import detector, bbox_merger, recognizer

def process_image(image_path):
    """
    Processes a single image and returns the recognized text as a JSON string.

    Args:
        image_path (str): Path to the input image.
        detector_model_file (str): Path to the detector model.
        recognizer_model_path (str): Path to the recognizer model.

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

    return recognizer_output

# (Optional) Example usage if you run main.py directly
if __name__ == "__main__":
    image_path = "files/input/Single Sample Inference/0001_front.jpg"
    detector_model_file = "checkpoints/FAST/fast_base_20240926-134200_epoch100.pt"  # Add this
    recognizer_model_path = "checkpoints/TrOCR"

    recognizer_output = process_image(
        image_path, detector_model_file, recognizer_model_path
    )

    print(recognizer_output)