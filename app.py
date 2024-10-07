import os
import json
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from main import process_image  # Import from main.py


import json
from pathlib import Path

from fastapi.middleware.cors import CORSMiddleware

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

@app.get("/load-json")
async def load_json():
    # Path to your JSON file
    json_file_path = Path("recog_data.json")

    # Load the JSON file
    try:
        with json_file_path.open() as f:
            data = json.load(f)
        return JSONResponse(content=data)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    try:
        file_location = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Call process_image to generate recog_data.json
        process_image(file_location)

        # Return the URL to access the uploaded image
        image_url = f"http://localhost:8000/uploads/{file.filename}"

        # Load the JSON file
        json_file_path = Path("recog_data.json")
        with json_file_path.open() as f:
            data = json.load(f)
        return JSONResponse(content={"image": image_url, "data":data})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)