import streamlit as st
import torch
import torchvision.models as models
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

# ===============================
# Page Config
# ===============================
st.set_page_config(
    page_title="Diabetic Retinopathy Detection",
    page_icon="🩺",
    layout="centered"
)

# ===============================
# Custom CSS (UI Design)
# ===============================
st.markdown("""
<style>

.main {
background: linear-gradient(135deg,#f5f7fa,#c3cfe2);
}

.title {
text-align:center;
font-size:45px;
font-weight:bold;
color:#2c3e50;
}

.subtitle{
text-align:center;
font-size:18px;
color:#555;
}

.upload-box{
border:2px dashed #4CAF50;
padding:25px;
border-radius:15px;
text-align:center;
background:white;
}

.result-box{
padding:20px;
border-radius:15px;
font-size:22px;
text-align:center;
font-weight:bold;
}

.footer{
text-align:center;
font-size:14px;
color:gray;
}

</style>
""", unsafe_allow_html=True)

# ===============================
# Load Model
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.efficientnet_b0(weights="DEFAULT")
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

model.load_state_dict(torch.load("best_efficientnet_eyepacs.pth", map_location=device))
model.to(device)
model.eval()

# ===============================
# Image Transform
# ===============================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# ===============================
# Header
# ===============================
st.markdown('<p class="title">🩺 Diabetic Retinopathy Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload a retina image to detect diabetic retinopathy using AI</p>', unsafe_allow_html=True)

st.write("")

# ===============================
# Upload Section
# ===============================
st.markdown('<div class="upload-box">', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload Retina Image",
    type=["jpg","png","jpeg"]
)

st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# Prediction
# ===============================
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Retina Image", use_container_width=True)

    img = transform(image).unsqueeze(0).to(device)

    with st.spinner("Analyzing Retina Image..."):
        with torch.no_grad():
            outputs = model(img)
            probs = F.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probs, 1)

    confidence = confidence.item() * 100
    prediction = prediction.item()

    st.subheader("Prediction Result")

    if prediction == 0:
        st.success(f"Healthy Eye Detected ✅")
        st.progress(confidence/100)
        st.write(f"Confidence: **{confidence:.2f}%**")

    else:
        st.error(f"Diabetic Retinopathy Detected ⚠️")
        st.progress(confidence/100)
        st.write(f"Confidence: **{confidence:.2f}%**")

# ===============================
# Footer
# ===============================
st.write("")
st.markdown('<div class="footer">AI Powered Retina Disease Detection</div>', unsafe_allow_html=True)
