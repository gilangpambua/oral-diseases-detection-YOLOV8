from pathlib import Path
from PIL import Image
import streamlit as st
from ultralytics import YOLO
import config

@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

# --- Page Config ---
st.set_page_config(
    page_title="Oral Diseases Detection With YOLOv8",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Title ---
st.title("Oral Diseases Detection With YOLOv8")

# --- Sidebar Config ---
st.sidebar.header("Model Configuration")
task_type = "Detection"

model = None
model_path = None

if task_type == "Detection":
    model_type = st.sidebar.selectbox("Select Model", config.DETECTION_MODEL_LIST)
    
    if model_type:
        model_path = Path(config.DETECTION_MODEL_DIR) / model_type
        if model_path.exists():
            try:
                model = load_model(model_path)
            except Exception as e:
                st.sidebar.error(f"Error loading model: {e}")
        else:
            st.sidebar.error(f"Model file not found: {model_path}")
else:
    st.error("Currently only 'Detection' function is implemented")

confidence = float(st.sidebar.slider("Select Model Confidence", 5, 100, 50)) / 100

# --- Image Upload ---
st.sidebar.header("Image")
source_img = st.sidebar.file_uploader("Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

col1, col2 = st.columns(2)

# --- Show Uploaded Image ---
with col1:
    if source_img:
        uploaded_image = Image.open(source_img)
        st.image(image=uploaded_image, caption="Uploaded Image", use_column_width=True)

# --- Prediction ---
if source_img:
    if st.button("Execution"):
        if model is None:
            st.error("Model is not loaded. Please select a valid model.")
        else:
            with st.spinner("Running..."):
                try:
                    res = model.predict(uploaded_image, conf=confidence)
                    boxes = res[0].boxes
                    res_plotted = res[0].plot()[:, :, ::-1]

                    with col2:
                        st.image(res_plotted, caption="Detected Image", use_column_width=True)

                    st.markdown("---")
                    st.subheader("Detection Results")

                    if not boxes:
                        st.write("No oral diseases were detected.")
                    else:
                        with st.expander("Detection Details"):
                            for box in boxes:
                                class_id = int(box.cls)
                                class_name = model.names.get(class_id, "Unknown")
                                conf_val = box.conf.item() * 100
                                x_center, y_center, width, height = [round(coord, 4) for coord in box.xywh.tolist()[0]]

                                st.write(f"**Class:** {class_name}")
                                st.write(f"**Confidence:** {conf_val:.2f}%")
                                st.write("**Bounding Box:**")
                                st.write(f"- X Center: {x_center}")
                                st.write(f"- Y Center: {y_center}")
                                st.write(f"- Width: {width}")
                                st.write(f"- Height: {height}")
                                st.write("---")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
