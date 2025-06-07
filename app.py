from pathlib import Path
from PIL import Image
import streamlit as st
from ultralytics import YOLO
import config

@st.cache_resource
def load_model(model_path):
    model = YOLO(model_path)
    return model

st.set_page_config(
    page_title="Oral Diseases Detection With YOLOV8",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Oral Diseases Detection With YOLOV8")

st.sidebar.header("Model Configuration")

task_type ="Detection"

model_type = None
if task_type == "Detection":
    model_type = st.sidebar.selectbox(
        "Select Model",
        config.DETECTION_MODEL_LIST
    )
else:
    st.error("Currently only 'Detection' function is implemented")

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 5, 100, 50)) / 100

model_path = ""
if model_type:
    model_path = Path(config.DETECTION_MODEL_DIR, str(model_type))
else:
    st.error("Please Select Model in Sidebar")

try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Unable to load model. Please check the specified path: {model_path}")

st.sidebar.header("Image")

source_img = st.sidebar.file_uploader(
    label="Choose an image...",
    type=("jpg", "jpeg", "png", 'bmp', 'webp')
)

col1, col2 = st.columns(2)

with col1:
    if source_img:
        uploaded_image = Image.open(source_img)
        st.image(
            image=uploaded_image,
            caption="Uploaded Image",
            use_column_width=True
        )

if source_img:
    if st.button("Execution"):
        with st.spinner("Running..."):
            res = model.predict(uploaded_image, conf=confidence)
            boxes = res[0].boxes
            res_plotted = res[0].plot()[:, :, ::-1]

            with col2:
                st.image(res_plotted,
                         caption="Detected Image",
                         use_column_width=True)
                
            st.markdown("---")
            st.subheader("Detection Results")

            if len(boxes) == 0:
                st.write("no oral diseases were detected")
            else:
                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            class_name = model.names[int(box.cls)]
                            confidence = box.conf.item() * 100
                            x_center, y_center, width, height = [round(coord, 4) for coord in box.xywh.tolist()[0]]

                            st.write(f"Class: {class_name}")
                            st.write(f"Confidence: {confidence:.2f}%")
                            st.write(f"Bounding Box Coordinates:")
                            st.write(f" - X Center: {x_center}")
                            st.write(f" - Y Center: {y_center}")
                            st.write(f" - Width: {width}") 
                            st.write(f" - Height: {height}")
                            st.write("---")
                except Exception as ex:
                    st.write("No image is uploaded yet!")