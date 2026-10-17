import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from tensorflow import keras

import config

st.set_page_config(page_title="EcoVision AI", page_icon="♻️", layout="centered")


@st.cache_resource
def load_model():
    return keras.models.load_model(str(config.MODEL_PATH))


def preprocess(image, img_size):
    img = image.convert("RGB").resize((img_size, img_size))
    arr = np.asarray(img, dtype="float32")
    return np.expand_dims(arr, axis=0)


st.title("♻️ EcoVision AI")
st.caption("Waste classification with MobileNetV2 transfer learning")

if not config.MODEL_PATH.exists():
    st.error(
        f"No trained model found at `{config.MODEL_PATH}`.\n\n"
        "Train one first:\n\n```\npython train.py\n```"
    )
    st.stop()

model = load_model()

uploaded = st.file_uploader("Upload a waste image", type=["jpg", "jpeg", "png", "webp"])

if uploaded is not None:
    image = Image.open(uploaded)
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Input image", use_container_width=True)

    with col2:
        probs = model.predict(preprocess(image, config.IMG_SIZE), verbose=0)[0]
        order = np.argsort(probs)[::-1]
        top = order[0]

        st.metric("Prediction", config.CLASS_LABELS[config.CLASS_NAMES[top]])
        st.write(f"Confidence: **{probs[top] * 100:.2f}%**")

    st.subheader("Class probabilities")
    chart_df = pd.DataFrame(
        {
            "Class": [config.CLASS_LABELS[c] for c in config.CLASS_NAMES],
            "Probability": probs,
        }
    ).sort_values("Probability", ascending=False)
    st.bar_chart(chart_df.set_index("Class"), horizontal=True)

    st.dataframe(
        chart_df.assign(Probability=lambda d: (d["Probability"] * 100).round(2)),
        hide_index=True,
        use_container_width=True,
    )
else:
    st.info("Upload an image to classify it into cardboard, glass, metal, paper, plastic, or trash.")
