import os
import json
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from main import process_image


app = FastAPI()
origins = [ "http://localhost", "http://localhost:3000","http://localhost:5173", "http://ocr_api.lamzingtech.com"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows all hosts
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Mount the static files directory
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

@app.post("/process/")
async def process_image_endpoint(
    image: UploadFile = File(...),
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

        # Process the image using the function from main.py
        recognized_text_json = process_image(temp_image_path)

        return {"recognized_text": recognized_text_json}
    except Exception as e:
        return {"error": str(e)}

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    try:
        file_location = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Return the URL to access the uploaded image
        image_url = f"https://localhost:8000/uploads/{file.filename}"

        json_file_path = Path("data.json")

    # Load the JSON file
        with json_file_path.open() as f:
            data = json.load(f)
        return JSONResponse(content={"image": image_url, "data":data})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)