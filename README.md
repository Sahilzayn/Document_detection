## Document Classification Model
This project classifies different types of documents (Bank Statement, PAN Card, Passport Card, Aadhar Card) using pre-trained deep learning models.

# Table of Contents
Introduction
Project Structure
Installation
Usage
Models
Contributing
License
Introduction
The goal of this project is to classify different types of documents using pre-trained deep learning models. The models are capable of identifying Bank Statements, PAN Cards, Passport Cards, and Aadhar Cards from given images.


# Project Title

A brief description of what this project does and who it's for.

## Project Directory Structure
data/: Contains the pre-trained models and sample images for testing.
notebooks/: Contains the Jupyter notebook for the project.
README.md: This README file.
requirements.txt: List of dependencies required for the project.
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/document-classification.git
cd document-classification
Install the required packages:

bash
Copy code
pip install -r requirements.txt
Usage
Place your models (.h5 files) in the data/ directory.

Place the images you want to classify in the data/sample_images directory.

Run the Jupyter notebook:

bash
Copy code
jupyter notebook notebooks/Detection_model.ipynb
Follow the instructions in the notebook to classify your images.

Alternatively, you can use the following Python script to classify an image:

python
Copy code
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained models
model1 = load_model('data/bank_statement_model.h5')
model2 = load_model('data/pan_card_model.h5')
model3 = load_model('data/passport_card_model.h5')
model4 = load_model('data/aadhar_card_model.h5')

# Function to preprocess the image
def preprocess_img(image_path):
    img = cv2.imread(image_path)
    resized_img = cv2.resize(img, (224, 224))
    normalized_img = resized_img.astype('float32') / 255.0
    return np.expand_dims(normalized_img, axis=0)

# Function to predict the image
def predict_image(image_path, model):
    processed_image = preprocess_img(image_path)
    predictions = model.predict(processed_image)
    return predictions

# Function to classify the document
def classify_document(image_path, doc_type):
    if doc_type == 'bank_statement':
        model = model1
        label = 'Bank Statement'
    elif doc_type == 'pan_card':
        model = model2
        label = 'PAN Card'
    elif doc_type == 'passport_card':
        model = model3
        label = 'Passport Card'
    elif doc_type == 'aadhar_card':
        model = model4
        label = 'Aadhar Card'
    else:
        raise ValueError("Invalid document type")

    predictions = predict_image(image_path, model)
    return label if predictions[0][0] >= 0.5 else f'Not a {label}'

# Example usage
image_path = 'data/sample_images/aadhar_card.jpg'
doc_type = 'aadhar_card'
result = classify_document(image_path, doc_type)
print(result)
Models
The project uses pre-trained models to classify documents. The models should be placed in the data/ directory and have the following names:

bank_statement_model.h5
pan_card_model.h5
passport_card_model.h5
aadhar_card_model.h5
Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

License
This project is licensed under the MIT License. See the LICENSE file for details
