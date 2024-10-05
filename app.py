from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from main import process_image  # Import the process_image function

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (replace with your frontend origin)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/process/")
async def process_image_endpoint(
    image: UploadFile = File(...),
    detector_model_file: str = "checkpoints/FAST/fast_base_20240926-134200_epoch100.pt",
    recognizer_model_path: str = "checkpoints/TrOCR",
):
    """
    Endpoint to process an uploaded image and return recognized text as JSON.
    """
    try:
        # Read the image contents
        image_contents = await image.read()

        # Save the image temporarily (you might want to use a temporary file)
        temp_image_path = "temp_image.jpg"
        with open(temp_image_path, "wb") as f:
            f.write(image_contents)

        # Process the image
        recognized_text_json = process_image(
            temp_image_path, detector_model_file, recognizer_model_path
        )

        return {"recognized_text": recognized_text_json}
    except Exception as e:
        return {"error": str(e)}