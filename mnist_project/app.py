import streamlit as st
import requests
from PIL import Image

API_URL = "http://127.0.0.1:8000/predict"

st.title("🖊️ MNIST Digit Recognition")
st.write("Upload a hand-written digit image (0–9).")

uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict Digit"):
        # Send to FastAPI
        files = {"file": uploaded_file.getvalue()}
        response = requests.post(API_URL, files=files)

        if response.status_code == 200:
            digit = response.json()["predicted_digit"]
            st.success(f"✅ Predicted Digit: **{digit}**")
        else:
            st.error("❌ Prediction failed.")
