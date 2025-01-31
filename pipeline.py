import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import requests
from io import BytesIO
import os
import logging

# ✅ Silence Unnecessary Logs
logging.getLogger("transformers").setLevel(logging.ERROR)

st.title("📸 AI-Powered Image Captioning System")

# ✅ Define Paths for Local Models
binary_classifier_path = "models/binary_classifier_resnet18.pth"  # Store in 'models/' directory
vit_gpt2_model_path = "models/vit_gpt2_captioning_model"  # Folder for ViT-GPT2 model

# ✅ Define ResNet-18 Binary Classifier Model
class PhotoClassifier(nn.Module):
    def __init__(self):
        super(PhotoClassifier, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)  # Binary Classification (1 output node)

    def forward(self, x):
        return self.model(x)

# ✅ Load the trained binary classifier
photo_classifier = PhotoClassifier()
photo_classifier.load_state_dict(torch.load(binary_classifier_path, map_location="cpu"))
photo_classifier.eval()  # Set to evaluation mode

# ✅ Load ViT-GPT2 Captioning Model from Local Folder
model = VisionEncoderDecoderModel.from_pretrained(vit_gpt2_model_path).to("cpu")
processor = ViTImageProcessor.from_pretrained(vit_gpt2_model_path)
tokenizer = AutoTokenizer.from_pretrained(vit_gpt2_model_path)

def load_image(url):
    """Load an image from a URL with error handling."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}  
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except requests.exceptions.RequestException as e:
        st.error(f"🚫 Image loading failed: {e}")
        return None

def is_photo(image):
    """Classify an image as Photo (1) or Not a Photo (0) using ResNet-18."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0)  

    with torch.no_grad():
        output = photo_classifier(image_tensor)
    
    return torch.sigmoid(output).item() > 0.5  

def generate_caption_vit_gpt2(image):
    """Generate a caption using ViT-GPT2 model."""
    inputs = processor(images=image, return_tensors="pt").to("cpu")
    with torch.no_grad():
        output = model.generate(**inputs)
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

# ✅ Upload Image or Use External URL
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])
image_url = st.text_input("Or enter an image URL:", "")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
elif image_url:
    image = load_image(image_url)
else:
    image = None

if image:
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ✅ Step 1: Classify Image as Photo or Not
    if is_photo(image):
        st.success("✅ This is a Photo. Proceeding to Captioning...")

        # ✅ Step 2: Generate Caption using ViT-GPT2
        caption = generate_caption_vit_gpt2(image)

        # ✅ Display Caption
        st.subheader("📝 Generated Caption:")
        st.write(caption)

        # ✅ Option to Download Caption
        st.download_button("Download Caption", caption, file_name="caption.txt")
    else:
        st.error("🚫 This is NOT a photo! Captioning will not be performed.")
        st.stop()  # ✅ Prevent further execution
