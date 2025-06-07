from pathlib import Path
from PIL import Image
import streamlit as st
from ultralytics import YOLO
import config

@st.cache_resource
def load_model(model_path):
    try:
        import torch
        from ultralytics.nn.tasks import DetectionModel
        torch.serialization.add_safe_globals({'ultralytics.nn.tasks.DetectionModel': DetectionModel})

        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Streamlit page config
st.set_page_config(
    page_title="Oral Diseases Detection With YOLOv8",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Oral Diseases Detection With YOLOv8")
st.sidebar.header("Model Configuration")

# Task type (only detection supported)
task_type = "Detection"

if task_type != "Detection":
    st.error("Currently only 'Detection' task is supported.")
    st.stop()

# Model selection
model_type = st.sidebar.selectbox("Select Model", config.DETECTION_MODEL_LIST)

# Confidence threshold
confidence = st.sidebar.slider("Select Model Confidence", 5, 100, 50) / 100

# Load model
model_path = str(Path(config.DETECTION_MODEL_DIR) / model_type)
model = load_model(model_path)

if not model:
    st.stop()

# Image upload
st.sidebar.header("Image Input")
source_img = st.sidebar.file_uploader("Choose an image...", type=("jpg", "jpeg", "png", "bmp", "webp"))

col1, col2 = st.columns(2)

# Display uploaded image
with col1:
    if source_img:
        uploaded_image = Image.open(source_img)
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

# Prediction & results
if source_img and st.button("Execution"):
    with st.spinner("Running model prediction..."):
        try:
            res = model.predict(uploaded_image, conf=confidence)
            boxes = res[0].boxes
            res_plotted = res[0].plot()[:, :, ::-1]

            with col2:
                st.image(res_plotted, caption="Detected Image", use_column_width=True)

            st.markdown("---")
            st.subheader("Detection Results")

            if len(boxes) == 0:
                st.info("No oral diseases were detected.")
            else:
                with st.expander("Detection Details"):
                    for box in boxes:
                        class_name = model.names[int(box.cls)]
                        conf = box.conf.item() * 100
                        x_center, y_center, width, height = [round(coord, 4) for coord in box.xywh.tolist()[0]]

                        st.write(f"**Class:** {class_name}")
                        st.write(f"**Confidence:** {conf:.2f}%")
                        st.write(f"**Bounding Box (XYWH):**")
                        st.write(f"- X Center: {x_center}")
                        st.write(f"- Y Center: {y_center}")
                        st.write(f"- Width: {width}")
                        st.write(f"- Height: {height}")
                        st.write("---")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
