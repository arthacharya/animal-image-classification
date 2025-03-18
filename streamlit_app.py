import streamlit as st
import torch
import json
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F

# Define the AnimalCNN class (must match your notebook)
class AnimalCNN(nn.Module):
    def __init__(self, num_classes):
        super(AnimalCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)  # conv layer 1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # conv layer 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # pooling layer
        
        # here i calculated the input size for the first fcn layer
        self._to_linear = None
        self._calculate_flattened_size()

        self.fc1 = nn.Linear(self._to_linear, 128)  # fcn layer 1
        self.fc2 = nn.Linear(128, num_classes)  # fcn layer 2 output

    def _calculate_flattened_size(self):
        with torch.no_grad():
            # here i did passed dummy input to know the size after convolutions and pooling
            dummy_input = torch.randn(1, 3, 128, 128)  # input size batch size = 1
            x = self.pool(F.relu(self.conv2(F.relu(self.conv1(dummy_input)))))
            self._to_linear = x.numel() 

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  #flatten tensor for fcn
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  #output layer
        return x

# Load configuration
@st.cache_resource
def load_config():
    with open("models/class_names.json", "r") as f:
        class_names = json.load(f)
    return class_names

@st.cache_resource
def load_model(model_name):
    class_names = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_name == "Custom CNN":
        model = AnimalCNN(num_classes=len(class_names))
        model.load_state_dict(torch.load("models/animal_cnn.pth", map_location=device))
    elif model_name == "Pretrained ResNet":
        model = models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
        model.load_state_dict(torch.load("models/pretrained_resnet.pth", map_location=device))
    
    model.eval()
    return model.to(device), class_names

def predict(image, model, device):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs, dim=1)
    return probs.cpu().numpy()[0]

def main():
    st.set_page_config(page_title="Animal Classifier", layout="wide")
    st.title("Animal Image Classification")
    
    # Sidebar controls
    st.sidebar.header("Model Selection")
    model_name = st.sidebar.selectbox("Choose Model", ["Custom CNN", "Pretrained ResNet"])
    
    # Load selected model
    model, class_names = load_model(model_name)
    device = next(model.parameters()).device
    
    # Main content
    uploaded_file = st.file_uploader("Upload an animal image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Classify"):
                probs = predict(image, model, device)
                pred_idx = np.argmax(probs)
                
                st.subheader("Prediction Result")
                st.metric(label="Predicted Class", value=class_names[pred_idx])
                st.metric(label="Confidence", value=f"{probs[pred_idx]*100:.2f}%")
        
        with col2:
            if 'probs' in locals():
                st.subheader("Class Confidence Distribution")
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.barplot(x=probs, y=class_names, ax=ax, palette="viridis")
                ax.set_xlabel("Confidence Score")
                ax.set_title("Prediction Probabilities")
                st.pyplot(fig)

if __name__ == "__main__":
    main()