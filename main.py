import shutil
from fastapi import FastAPI, UploadFile, Form, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image

app = FastAPI()

# Mount the 'static' directory to serve static files (e.g., CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates for HTML rendering
templates = Jinja2Templates(directory="templates")

# Directory to store uploaded images
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Define a function to save uploaded files
def save_uploaded_file(file, destination):
    with open(destination, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

# Load your machine learning models outside the route functions
model1 = tf.keras.models.load_model("model-12-0.97-0.13.h5")
model2 = tf.keras.models.load_model("tb_classifier.h5")
model3 = tf.keras.models.load_model("pneu_classifier.h5")

def preprocess_image(image_path, target_size=(512, 512)):
    img = image.load_img(image_path, target_size=target_size)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize the image
    return img

# Define the main route
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Define the route to handle image uploads
@app.post("/upload/")
async def upload_file(
    request: Request,
    choice: int = Form(...),
    data: UploadFile = Form(...),):
    file_location = os.path.join(UPLOAD_DIR, 'internet.jpg')
    save_uploaded_file(data, file_location)

    if choice == 1:
        # Make a prediction using model1
        img = preprocess_image(file_location, target_size=(200, 200))
        predictions = model1.predict(img)
        predicted_class = np.argmax(predictions)
        class_labels = ["glioma", "meningioma", "notumor", "pituitary"]
        result = class_labels[predicted_class]
        return result

    elif choice == 2:
        # Make a prediction using model2
        img = preprocess_image(file_location, target_size=(512, 512))
        prediction = model2.predict(img)
        print(prediction[0][0])
        if prediction <= 0.25:
            result = "Normal"
        else:
            result = "Tuberculosis"
        return result

    elif choice == 3:
        # Make a prediction using model3
        img = preprocess_image(file_location, target_size=(512, 512))
        prediction = model3.predict(img)
        print(prediction[0][0])
        if prediction <= 0.5:
            result = "Normal"
        else:
            result = "PNEUMONIA"
        return result

    # If none of the above conditions match, return a response
    return "Invalid choice or prediction failed."

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
