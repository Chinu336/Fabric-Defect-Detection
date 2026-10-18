from pathlib import Path
import json
import joblib
import numpy as np
import streamlit as st
from texture_features import extract_texture_features, display_grayscale

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "fabric_model.joblib"
REPORT_PATH = ROOT / "reports" / "evaluation.json"

st.set_page_config(page_title="Fabric Defect Detection", page_icon="🧵", layout="wide")
st.title("🧵 Fabric Defect Detection Using Texture Analysis")
st.write(
    "Upload a fabric image. The app extracts GLCM, LBP and Gabor texture features "
    "and classifies the image with a Random Forest model."
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.warning("The trained model was not found. Add dataset images and run `python train_model.py` first.")
    st.stop()

model = load_model()
with st.sidebar:
    st.header("About")
    st.write("**Features:** GLCM + LBP + Gabor")
    st.write("**Classifier:** Random Forest")
    st.write("**Classes:** Normal / Defective")
    st.caption("Academic prototype. Predictions are not a guarantee of industrial quality.")

uploaded = st.file_uploader("Upload a fabric image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
if uploaded is not None:
    try:
        preview = display_grayscale(uploaded)
        uploaded.seek(0)
        features = extract_texture_features(uploaded)
        uploaded.seek(0)
        if features.shape[0] != model.n_features_in_:
            st.error("Feature mismatch. Retrain with the included train_model.py.")
            st.stop()

        probabilities = model.predict_proba(features.reshape(1, -1))[0]
        predicted_label = int(model.classes_[int(np.argmax(probabilities))])
        names = {0: "Normal", 1: "Defective"}
        confidence = float(np.max(probabilities)) * 100

        left, right = st.columns(2)
        with left:
            st.subheader("Input image")
            st.image(preview, use_container_width=True)
        with right:
            st.subheader("Prediction")
            if predicted_label == 1:
                st.error("⚠️ Defective fabric predicted")
            else:
                st.success("✅ Normal fabric predicted")
            st.metric("Model confidence", f"{confidence:.2f}%")
            st.write(f"**Predicted class:** {names[predicted_label]}")
            st.caption("Probability is the model's estimate, not a guarantee of correctness.")
            st.subheader("Class probabilities")
            for class_id, probability in zip(model.classes_, probabilities):
                st.write(f"{names[int(class_id)]}: {probability * 100:.2f}%")
                st.progress(float(probability))

        names_of_features = [
            "GLCM contrast mean", "GLCM contrast std",
            "GLCM dissimilarity mean", "GLCM dissimilarity std",
            "GLCM homogeneity mean", "GLCM homogeneity std",
            "GLCM energy mean", "GLCM energy std",
            "GLCM correlation mean", "GLCM correlation std",
            "GLCM ASM mean", "GLCM ASM std",
        ]
        st.subheader("Extracted texture measurements")
        st.json({
            name: float(value)
            for name, value in zip(names_of_features, features[:len(names_of_features)])
        })
    except Exception as exc:
        st.error(f"Could not analyze this image: {exc}")

st.divider()
st.subheader("Model evaluation")
if REPORT_PATH.exists():
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        report = json.load(f)
    c1, c2, c3 = st.columns(3)
    c1.metric("Test accuracy", f"{report['accuracy'] * 100:.2f}%")
    c2.metric("Macro F1 score", f"{report['macro_f1']:.3f}")
    c3.metric("Test images", report["test_images"])
    st.write("Confusion matrix (rows = actual; columns = predicted):")
    st.dataframe(report["confusion_matrix"], use_container_width=True)
    st.write("Classification report")
    st.json(report["classification_report"])
else:
    st.info("Train the model to display evaluation metrics.")

st.caption("Academic prototype. Results depend on dataset quality, variety and labeling.")
