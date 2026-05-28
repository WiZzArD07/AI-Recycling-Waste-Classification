import requests
import streamlit as st
from PIL import Image

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(

    page_title="AI Waste Classification",

    page_icon="♻️",

    layout="centered"
)

# =====================================================
# TITLE
# =====================================================

st.title("♻️ AI Waste Classification System")

st.write(
    "Upload a waste image and the AI will classify it."
)

# =====================================================
# IMAGE UPLOADER
# =====================================================

uploaded_file = st.file_uploader(

    "Upload Waste Image",

    type=["jpg", "jpeg", "png"]
)

# =====================================================
# PROCESS IMAGE
# =====================================================

if uploaded_file is not None:

    # Display uploaded image
    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # =================================================
    # PREDICT BUTTON
    # =================================================

    if st.button("Predict Waste Type"):

        with st.spinner("Analyzing Image..."):

            # Send image to API
            files = {

                "file": uploaded_file.getvalue()
            }

            response = requests.post(

                "http://127.0.0.1:8000/predict",

                files=files
            )

            # Get JSON response
            result = response.json()

            # =================================================
            # DISPLAY RESULTS
            # =================================================

            st.success("Prediction Complete!")

            st.subheader("Prediction")

            st.write(
                f"### {result['prediction'].upper()}"
            )

            st.subheader("Confidence")

            st.write(
                f"### {result['confidence']}%"
            )