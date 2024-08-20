import streamlit as st
import numpy as np
import cv2

# Set up the header and main containers
header = st.container()
resizer = st.container()

# Header section
with header:
    st.title('Image Processing with Streamlit')

# Image processing section
with resizer:
    # File uploader
    uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'webp', 'png'])
    
    if uploaded_file is not None:
        # Convert the file to an OpenCV image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        # Display the original image
        st.image(img, channels="BGR", caption="Uploaded Image", use_column_width=True)

        st.write("Resize the Image")
        
        # Sliders for resizing
        col1, col2 = st.columns(2)
        with col1:
            width = st.slider("Select width:", min_value=50, max_value=img.shape[1], value=img.shape[1])
        with col2:
            height = st.slider("Select height:", min_value=50, max_value=img.shape[0], value=img.shape[0])
        
        # Resize using selected dimensions
        resized_img = cv2.resize(img, (width, height))
        
        # Display resized image
        st.image(resized_img, channels="BGR", caption="Resized Image", use_column_width=True)
        
        # Buttons for additional processing
        
        if st.button("Resize to 1080x720"):
            resized_img = cv2.resize(img, (1080, 720))
            st.image(resized_img, channels="BGR", caption="Resized Image (1080x720)", use_column_width=True)
        st.write("Apply additional processing:")
        if st.button("Convert to Grayscale"):
            gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
            st.image(gray, caption="Grayscale Image", use_column_width=True)
        
        kernel_size = st.slider("Select Blur Kernel Size:", min_value=1, max_value=50, value=15, step=2)
        if st.button("Apply Blur"):
            if kernel_size % 2 == 0:  # Ensure kernel size is odd
                kernel_size += 1
            blur = cv2.GaussianBlur(resized_img, (kernel_size, kernel_size), 0)
            st.image(blur, channels="BGR", caption="Blur Image", use_column_width=True)

        st.write("Adjust Canny Edge Detection Parameters:")
        upper_thresh = st.slider("Upper Threshold:", min_value=0, max_value=255, value=150)
        lower_thresh = st.slider("Lower Threshold:", min_value=0, max_value=255, value=50)

        if st.button("Apply Edge Detection"):
            gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, lower_thresh, upper_thresh)
            st.image(edges, caption="Edge Detection Image", use_column_width=True)
