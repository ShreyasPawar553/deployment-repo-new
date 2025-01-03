import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import gdown

# Initialize FastAPI app
app = FastAPI()

# Working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

# Google Drive file ID for your model
file_id = '18tfFcyf9DgK_r7cPIMPDDmPB7IT0-NIp'
model_path = "plant_disease_prediction_model.h5"

# Download the model if it doesn't exist
if not os.path.exists(model_path):
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, model_path, quiet=False)

# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# Load the class names
with open(os.path.join(working_dir, "class_indices.json"), "r") as f:
    class_indices = json.load(f)

# Image preprocessing function
def load_and_preprocess_image(image, target_size=(224, 224)):
    img = Image.open(BytesIO(image))
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

# Prediction function
def predict_image_class(model, image, class_indices):
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# FastAPI endpoint to classify an uploaded image
@app.post("/classify/")
async def classify_image(file: UploadFile = File(...)):
    # Read the image from the uploaded file
    image_data = await file.read()
    
    # Get prediction
    prediction = predict_image_class(model, image_data, class_indices)
    
    # Return the prediction as a JSON response
    return JSONResponse(content={"prediction": prediction})

