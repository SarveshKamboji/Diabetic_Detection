import streamlit as st
import torch
import torchvision.models as models
import torch.nn as nn
from PIL import Image
from torchvision import transforms

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
# UI
# ===============================
st.title("Diabetic Retinopathy Detection")

st.write("Upload a retina image to detect diabetic retinopathy")

uploaded_file = st.file_uploader("Upload Retina Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        prediction = torch.argmax(outputs,1).item()

    if prediction == 0:
        st.success("Healthy Eye Detected")
    else:
        st.error("Diabetic Retinopathy Detected")