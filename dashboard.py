import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
from predict import predict_image
from PIL import Image
import pandas as pd
from heatmap import generate_heatmap
import cv2
import numpy as np




st.title("Diabetic Retinopathy Detection Dashboard")


uploaded_file = st.file_uploader("Upload Retinal Image")

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Retina Image")

    label, confidence = predict_image(uploaded_file)

    st.write("Prediction:", label)
    st.write("Confidence:", round(confidence * 100, 2), "%")

    # Generate heatmap
    heatmap = generate_heatmap(uploaded_file)

    img = np.array(image.resize((224, 224)))

    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    superimposed_img = heatmap * 0.4 + img

    st.subheader("Retina Damage Heatmap")

    st.image(superimposed_img.astype("uint8"))

# Example dataset analytics

data = {
    "Stage": ["No DR", "Mild", "Moderate", "Severe", "Proliferative"],
    "Count": [200, 80, 120, 60, 40]
}

df = pd.DataFrame(data)

fig = px.bar(df, x="Stage", y="Count", title="DR Distribution")

st.plotly_chart(fig)
# Sample dataset
data = {
    "Stage": ["No DR","Mild","Moderate","Severe","Proliferative"],
    "Image_Count": [200,80,120,60,40],
    "Severity_Score": [0,1,2,3,4]
}

df = pd.DataFrame(data)
st.subheader("DR Severity Heatmap")

corr = df.corr(numeric_only=True)

fig, ax = plt.subplots()

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    linewidths=0.5,
    ax=ax
)

st.pyplot(fig)
heatmap_option = st.selectbox(
    "Select Heatmap Type",
    ["Correlation Heatmap", "Severity Heatmap", "Dataset Distribution Heatmap"]
)
data = {
    "Stage": ["No DR","Mild","Moderate","Severe","Proliferative"],
    "Image_Count": [200,80,120,60,40],
    "Severity_Score": [0,1,2,3,4],
    "Lesion_Count":[10,25,50,80,120]
}

df = pd.DataFrame(data)
fig, ax = plt.subplots()

if heatmap_option == "Correlation Heatmap":

    corr = df.corr(numeric_only=True)

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        linewidths=0.5,
        ax=ax
    )

elif heatmap_option == "Severity Heatmap":

    severity_data = df.pivot_table(
        values="Lesion_Count",
        index="Stage"
    )

    sns.heatmap(
        severity_data,
        annot=True,
        cmap="Reds",
        ax=ax
    )

elif heatmap_option == "Dataset Distribution Heatmap":

    dist_data = df.pivot_table(
        values="Image_Count",
        index="Stage"
    )

    sns.heatmap(
        dist_data,
        annot=True,
        cmap="Blues",
        ax=ax
    )

st.pyplot(fig)