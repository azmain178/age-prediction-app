# Import non-streamlit packages first
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import xgboost as xgb
from io import BytesIO
import traceback

# Import streamlit last
import streamlit as st

# Must be the very first Streamlit command
st.set_page_config(
    page_title="Age Prediction",
    page_icon="👤",
    layout="wide"
)

# Enable debug mode
debug = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if debug:
    st.write(f"Using device: {device}")

# Set up image transformations
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@st.cache_resource
def load_models():
    try:
        if debug:
            st.write("Loading models...")
        
        # Load face model
        face_model = models.resnet50(pretrained=False)
        face_model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(face_model.fc.in_features, 1)
        )
        if debug:
            st.write("Loading face_model.pth...")
        face_model.load_state_dict(torch.load('face_model.pth', map_location=device))
        face_model = face_model.to(device)
        face_model.eval()

        # Load other models
        if debug:
            st.write("Loading biomarker models...")
        bio_model = xgb.XGBRegressor()
        bio_model.load_model('bio_model.json')

        face_adjuster = xgb.XGBRegressor()
        face_adjuster.load_model('face_adjuster.json')

        stack_model = xgb.XGBRegressor()
        stack_model.load_model('stack_model.json')

        if debug:
            st.write("All models loaded successfully!")
        return face_model, bio_model, face_adjuster, stack_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        raise

def preprocess_image(image_data):
    try:
        image = Image.open(BytesIO(image_data)).convert("RGB")
        if debug:
            st.write(f"Image size before resize: {image.size}")
        image = val_transforms(image)
        if debug:
            st.write(f"Tensor shape after transform: {image.shape}")
        return image.unsqueeze(0).to(device)
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        raise

def preprocess_biomarkers(biomarkers_dict):
    try:
        required_keys = ['Height (cm) ', 'Weight (kg)', 'BMI', 'Blood Oxygen', 
                        'Blood Sugar(mg/dl)', 'Systolic_BP', 'Diastolic_BP']
        bio_data = pd.DataFrame([biomarkers_dict], columns=required_keys)
        if debug:
            st.write("Biomarker data:")
            st.write(bio_data)
        return bio_data
    except Exception as e:
        st.error(f"Error preprocessing biomarkers: {str(e)}")
        raise

def predict_age(image, biomarkers_dict):
    try:
        # Load models (they will be cached by Streamlit)
        face_model, bio_model, face_adjuster, stack_model = load_models()
        
        # Preprocess inputs
        image_tensor = preprocess_image(image)
        bio_data = preprocess_biomarkers(biomarkers_dict)

        # Face prediction
        with torch.no_grad():
            face_pred = face_model(image_tensor).cpu().numpy().flatten()[0]
            if debug:
                st.write(f"Raw face prediction: {face_pred}")
        
        face_pred_adj = face_adjuster.predict(np.array([[face_pred]]))[0]
        if debug:
            st.write(f"Adjusted face prediction: {face_pred_adj}")

        # Biomarkers prediction
        bio_pred = bio_model.predict(bio_data)[0]
        if debug:
            st.write(f"Biomarkers prediction: {bio_pred}")

        # Hybrid prediction
        stack_input = np.column_stack((face_pred_adj, bio_pred))
        hybrid_pred = stack_model.predict(stack_input)[0]
        if debug:
            st.write(f"Final hybrid prediction: {hybrid_pred}")

        return hybrid_pred
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        raise

# Streamlit UI
st.title("Age Prediction from Face Image and Biomarkers")
st.write("Upload a face image and enter biomarker data to predict age")

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.subheader("Face Image")
    uploaded_file = st.file_uploader("Choose a face image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Read the file once and store its contents
        image_bytes = uploaded_file.read()
        # Display the image using the bytes
        st.image(BytesIO(image_bytes), caption="Uploaded Image", use_column_width=True)

with col2:
    st.subheader("Biomarker Data")
    height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0)
    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=24.0)
    blood_oxygen = st.number_input("Blood Oxygen", min_value=80.0, max_value=100.0, value=98.0)
    blood_sugar = st.number_input("Blood Sugar (mg/dl)", min_value=70.0, max_value=400.0, value=100.0)
    systolic_bp = st.number_input("Systolic BP", min_value=80.0, max_value=200.0, value=120.0)
    diastolic_bp = st.number_input("Diastolic BP", min_value=40.0, max_value=130.0, value=80.0)

# Predict button
if st.button("Predict Age"):
    if uploaded_file is None:
        st.error("Please upload a face image")
    else:
        try:
            # Create biomarkers dictionary
            biomarkers = {
                'Height (cm) ': height,
                'Weight (kg)': weight,
                'BMI': bmi,
                'Blood Oxygen': blood_oxygen,
                'Blood Sugar(mg/dl)': blood_sugar,
                'Systolic_BP': systolic_bp,
                'Diastolic_BP': diastolic_bp
            }
            
            # Make prediction using the stored image bytes
            with st.spinner("Predicting age..."):
                predicted_age = predict_age(image_bytes, biomarkers)
                
            # Display result
            st.success(f"Predicted Age: {predicted_age:.1f} years")
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            if debug:
                st.error(f"Full error details: {str(e)}")
                st.error(traceback.format_exc())