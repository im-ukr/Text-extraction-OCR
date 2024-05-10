import streamlit as st
import cv2 as cv
import pytesseract
import numpy as np
import time

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract"

def extract_text(image):
    text = pytesseract.image_to_string(image)
    return text

def preprocess_image(image, grayscale, crop_area):
    if grayscale:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    if crop_area:
        x, y, w, h = crop_area
        image = image[y:y+h, x:x+w]
    return image

st.title('Image Text Extraction WebApp')

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    file_bytes = uploaded_image.read()
    image = cv.imdecode(np.frombuffer(file_bytes, np.uint8), 1)
    
    st.sidebar.title('Image Preprocessing')
    grayscale = st.sidebar.checkbox('Grayscale')
    crop = st.sidebar.checkbox('Crop')
    
    if crop:
        st.sidebar.write("Select the crop area by adjusting the sliders.")
        x = st.sidebar.slider('X', 0, image.shape[1], 0)
        y = st.sidebar.slider('Y', 0, image.shape[0], 0)
        w = st.sidebar.slider('Width', 0, image.shape[1], image.shape[1])
        h = st.sidebar.slider('Height', 0, image.shape[0], image.shape[0])
        crop_area = (x, y, w, h)
    else:
        crop_area = None

    image = preprocess_image(image, grayscale, crop_area)
    
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button('Extract Text'):
        progress_bar = st.progress(0)
        progress_message = st.empty()
        progress = 0
        while progress <= 100:
            progress_message.text(f'Extracting text... {progress}%')
            progress_bar.progress(progress)
            progress += 80
            time.sleep(1)
        progress_message.text('Extracting text... Done!\nDisplaying result...')
        progress_bar.empty()
        extracted_text = extract_text(image)
        st.write("### Extracted Text:")
        st.write(extracted_text)
