from fastapi import FastAPI, File, UploadFile, Request, status, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid
from ultralytics import YOLO
import numpy as np
import cv2
import os
import time
from functools import wraps



app = FastAPI()

'''
    Allow CORS
'''
origins = [
    "http://localhost:3000",  # React
    "http://localhost:8080",  # Vue.js
    "http://localhost:8000",  # Angular
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


imageDirectory = "uploadedFile" # store uploaded image in this folder

if not os.path.exists(imageDirectory):
    os.makedirs(imageDirectory)

model = YOLO("best (3).pt")

def rate_limited(max_calls: int, time_frame: int):
    def decorator(func):
        calls = []

        @wraps(func)
        async def wrapper(*args, **kwargs):
            now = time.time()
            calls_in_time_frame = [call for call in calls if call > now - time_frame]
            if len(calls_in_time_frame) >= max_calls:
                raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded!")
            calls.append(now)
            return await func(*args, *kwargs)

        return wrapper

    return decorator


def objectDetector(filename):
    print(f"=========================Ini filename: {filename}")
    results = model.predict(source=f"{imageDirectory}/{filename}", show=True, conf=0.5, save = True, exist_ok=True)
    
    imagePath = "runs/detect/predict/" + filename

    return imagePath

@app.get("/")
@rate_limited(max_calls=10, time_frame=30) # decorator to limit request
async def index():
    return {"message": "Hellow World"}


@app.post("/upload")
#@rate_limited(max_calls=100, time_frame=60) # decorator to limit request
async def uploadFile(file: UploadFile = File(...)):
    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()

    #save the file
    with open(f"{imageDirectory}/{file.filename}", "wb") as f:
        f.write(contents)

    imagePath = objectDetector(file.filename)
    return FileResponse(imagePath)

@app.post("/uploadFileBase64")
#@rate_limited(max_calls=100, time_frame=60) # decorator to limit request
async def uploadFileBase64(request: Request):
    try:
        data = await request.json()
        base64_image = data.get('image')
        if not base64_image:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No image provided")

        # Decode the base64 image
        try:
            image_data = base64.b64decode(base64_image)
            np_image = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        except Exception as e:
            logging.error(f"Error decoding base64 image: {e}")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid base64 image")

        # Save the decoded image
        filename = f"{uuid.uuid4()}.jpg"
        filepath = f"{imageDirectory}/{filename}"
        cv2.imwrite(filepath, img)

        imagePath = objectDetector(filename)
        return FileResponse(imagePath)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal Server Error")

@app.get("/detectedImage")
# @rate_limited(max_calls=100, time_frame=60) # decorator to limit request
async def showImage():
    if (os.path.exists("result.png")):
        imagePath = "result.png"
        return FileResponse(imagePath)
    else:
        return {"status", "error"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="info") # adjust port
