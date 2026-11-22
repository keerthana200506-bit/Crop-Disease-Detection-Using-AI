import streamlit as st
from PIL import Image
from predict import load_model_and_classes, predict_image

st.set_page_config(page_title="Crop Disease Detection", page_icon="🌿", layout="centered")

st.title("🌿 Crop Disease Detection Using AI")
st.write(
    "Upload a crop leaf image to view the model's predicted class. "
    "This is an experimental classifier and should not be used as the sole basis "
    "for crop-treatment decisions."
)

@st.cache_resource
def get_model():
    return load_model_and_classes()

try:
    model, class_names = get_model()
except Exception as exc:
    st.warning(
        "The trained model is not ready yet. Prepare the dataset and run "
        "`python train.py` before using predictions."
    )
    st.code("python train.py --data_dir data --epochs 10 --batch_size 16")
    with st.expander("Technical details"):
        st.text(str(exc))
    st.stop()

st.caption("Recognized classes: " + ", ".join(class_names))
uploaded = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded leaf image", use_container_width=True)
    if st.button("Analyze image", type="primary"):
        with st.spinner("Analyzing image..."):
            results = predict_image(image, model, class_names)
        best = results[0]
        st.subheader("Prediction")
        st.success(f"{best['class']} — {best['confidence']:.1%} confidence")
        st.write("Top predictions")
        for item in results:
            st.write(f"**{item['class']}**")
            st.progress(min(1.0, max(0.0, item["confidence"])))
            st.caption(f"{item['confidence']:.1%}")
        st.info(
            "Confidence is not a guarantee of correctness. Image quality, "
            "lighting, crop variety and field conditions can affect results."
        )
