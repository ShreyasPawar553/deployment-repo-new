import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import gdown
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Initialize FastAPI app
app = FastAPI()

# ================== PLANT DISEASE DETECTION ==================

# Model paths
MODEL_PATH = "plant_disease_prediction_model.h5"
CLASS_INDICES_PATH = "class_indices.json"

# Google Drive file ID
file_id = '18tfFcyf9DgK_r7cPIMPDDmPB7IT0-NIp'

# Download model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    gdown.download(f'https://drive.google.com/uc?id={file_id}', MODEL_PATH, quiet=False)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Load class indices
if os.path.exists(CLASS_INDICES_PATH):
    with open(CLASS_INDICES_PATH, "r") as f:
        class_indices = json.load(f)
else:
    class_indices = {}

# Image preprocessing function
def preprocess_image(image, target_size=(224, 224)):
    img = Image.open(BytesIO(image)).resize(target_size)
    img_array = np.expand_dims(np.array(img).astype('float32') / 255.0, axis=0)
    return img_array

# Prediction function
def predict_plant_disease(image):
    preprocessed_img = preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    return class_indices.get(str(predicted_class_index), "Unknown Class")

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    image_data = await file.read()
    prediction = predict_plant_disease(image_data)
    return JSONResponse(content={"prediction": prediction})

# ================== CROP & FERTILIZER RECOMMENDATION ==================

# Load models and scalers
fertilizer_model = pickle.load(open('classifier.pkl', 'rb'))
fertilizer_info = pickle.load(open('fertilizer.pkl', 'rb'))
crop_model = pickle.load(open('model.pkl', 'rb'))
scaler_standard = pickle.load(open('standscaler.pkl', 'rb'))
scaler_minmax = pickle.load(open('minmaxscaler.pkl', 'rb'))

@app.post("/fertilizer/predict")
async def predict_fertilizer(data: dict):
    try:
        input_data = np.array([[data["temp"], data["humi"], data["mois"], data["soil"], data["crop"], data["nitro"], data["pota"], data["phosp"]]])
        prediction_idx = fertilizer_model.predict(input_data)[0]
        result_label = fertilizer_info.classes_[prediction_idx] if hasattr(fertilizer_info, 'classes_') else 'Unknown'
        return JSONResponse(content={"fertilizer": result_label})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

@app.post("/crop/predict")
async def predict_crop(data: dict):
    try:
        feature_list = np.array([[data["N"], data["P"], data["K"], data["temp"], data["humidity"], data["ph"], data["rainfall"]]])
        scaled_features = scaler_minmax.transform(feature_list)
        final_features = scaler_standard.transform(scaled_features)
        prediction = crop_model.predict(final_features)[0]
        crop_dict = {
            1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya",
            7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes",
            12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil", 16: "Blackgram",
            17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas", 20: "Kidneybeans",
            21: "Chickpea", 22: "Coffee"
        }
        result = crop_dict.get(int(prediction), 'Unknown crop')
        return JSONResponse(content={"recommended_crop": result})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
