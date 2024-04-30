import torch
from torchvision import models, transforms, datasets
import torch.nn.functional as F
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import os
import streamlit as st 

st.header('Flower Classification using Pytorch and CNN')

import os
class_names = list()
for val in os.listdir('images/'): 
    class_names.append(val)

# Load the saved model
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 1000)  # Adjust to match the original model's output units
model.load_state_dict(torch.load('flower_classification_model.pth'))
model.eval()


with st.expander("About --- "):
    st.write("This model can detect Few images")
    for i in range(len(class_names)):
        st.write(f"{i+1}. {class_names[i]}")
        
uploaded_file = st.file_uploader('Upload an Image', type=('jpg','png','jpeg'))

def classify_images(image_path):
    # Load the saved model
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 1000)  # Adjust to match the original model's output units
    model.load_state_dict(torch.load('flower_classification_model.pth'))
    model.eval()
    
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        output = model(input_batch)

    # Get the predicted class
    _, predicted_class = output.max(1)

    # Map the predicted class to the class name
    # class_names = ['daisy', 'dandelion']  # Make sure these class names match your training data
    predicted_class_name = class_names[predicted_class.item()]

    # Get class probabilities
    probabilities = F.softmax(output[0], dim=0)
    # Load class labels (assuming you have them)
    flower_names = class_names

    # Get the predicted class index and its probability
    predicted_class_idx = torch.argmax(probabilities).item()
    predicted_probability = probabilities[predicted_class_idx].item()

    # Generate the outcome string
    if (predicted_probability*100) > 70:
        outcome = f"The predicted class is: {predicted_class_name} with \nThe image belongs to {class_names[predicted_class_idx]} with a score of {predicted_probability*100:.2f}%."
    else:
        outcome = f"It may be as prediction: {predicted_class_name} with \n Because The image belongs to {class_names[predicted_class_idx]} with a score of {predicted_probability*100:.2f}%"
    return outcome

if uploaded_file is not None:
    print(uploaded_file)
    st.image(uploaded_file, width = 200)
    st.markdown(classify_images(uploaded_file))