import json
from modules import detector, bbox_merger, recognizer, layloutlm

def load_models():
    detector_model_file = "checkpoints/FAST/fast_base_20240926-134200_epoch100.pt"
    recognizer_model_path = "checkpoints/TrOCR"
    layoutlm_model_path = "checkpoints/LayoutLM"

    trained_detector, default_detector = detector.load_detector(detector_model_file)
    trocr_processor, trocr_model, device = recognizer.load_recognizer(recognizer_model_path)
    llm_model, llm_processor = layloutlm.load_layoutlm(layoutlm_model_path)

    detector_models = [trained_detector, default_detector]
    recognizer_models = [trocr_processor, trocr_model, device]
    layoutlm_models = [llm_model, llm_processor]

    models = [detector_models, recognizer_models, layoutlm_models]
    return models

def process_image(image_path, models):
    # 1. Detect words
    detector_models = models[0]
    doctr_coco, doctr_result = detector.detect_words(image_path, detector_models, model="trained")

    # 2. Merge bounding boxes
    # box_merged_coco = bbox_merger.merge_annotations(doctr_coco)

    # 3. Recognize text
    recogniser_models = models[1]
    trocr_result, trocr_coco = recognizer.recognize_text(doctr_coco, image_path, recogniser_models)

    # 4. Recognise Layout 
    layoutlm_models = models[2]
    layoutlm_result = layloutlm.assign_tag(trocr_coco, image_path, layoutlm_models)

    return layoutlm_result

# (Optional) Example usage if you run main.py directly
if __name__ == "__main__":
    image_path = "files/input/1.jpg"

    models = load_models()

    layoutlm_result = process_image(image_path, models)
    print(layoutlm_result)

    with open("files/input/final_result.json", "w") as file:
        json.dump(layoutlm_result, file)