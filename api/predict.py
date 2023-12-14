from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import io
import torch.nn as nn
import torch.nn.functional as F

from flask_cors import CORS

index_to_label = {0: 'Raspberry___healthy', 1: 'Strawberry___Leaf_scorch', 2: 'Tomato___Leaf_Mold', 3: 'Apple___Cedar_apple_rust', 4: 'Corn_(maize)___healthy', 5: 'Tomato___Target_Spot', 6: 'Grape___Esca_(Black_Measles)', 7: 'Squash___Powdery_mildew', 8: 'Corn_(maize)___Common_rust_', 9: 'Tomato___Early_blight', 10: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 11: 'Grape___healthy', 12: 'Peach___healthy', 13: 'Blueberry___healthy', 14: 'Apple___Apple_scab', 15: 'Orange___Haunglongbing_(Citrus_greening)', 16: 'Tomato___Spider_mites Two-spotted_spider_mite', 17: 'Potato___Late_blight', 18: 'Tomato___Tomato_mosaic_virus', 19: 'Apple___healthy', 20: 'Soybean___healthy', 21: 'Peach___Bacterial_spot', 22: 'Pepper,_bell___healthy', 23: 'Tomato___healthy', 24: 'Corn_(maize)___Northern_Leaf_Blight', 25: 'Potato___healthy', 26: 'Tomato___Late_blight', 27: 'Tomato___Bacterial_spot', 28: 'Apple___Black_rot', 29: 'Grape___Black_rot', 30: 'Cherry_(including_sour)___Powdery_mildew', 31: 'Strawberry___healthy', 32: 'Tomato___Septoria_leaf_spot', 33: 'Pepper,_bell___Bacterial_spot', 34: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 35: 'Potato___Early_blight', 36: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 37: 'Cherry_(including_sour)___healthy'}
index_to_description = {
    0: 'Raspberry, a perennial fruit plant, is healthy with no signs of disease or distress.',
    1: 'Strawberry Leaf Scorch: A condition in strawberry plants where leaves develop red-purple spots and blight due to a fungal infection.',
    2: 'Tomato Leaf Mold: A fungal disease in tomatoes, characterized by pale green or yellow spots on leaves, leading to leaf curling and wilting.',
    3: 'Apple Cedar Apple Rust: A fungal disease in apple trees causing bright orange or yellow circular spots on leaves, leading to leaf drop.',
    4: 'Healthy Corn (Maize): Vibrant and robust corn plants, free from visible diseases, showing good growth and development.',
    5: 'Tomato Target Spot: A fungal disease causing concentric rings on tomato leaves and fruits, leading to significant yield loss.',
    6: 'Grape Esca (Black Measles): A fungal disease in grapevines, marked by tiger-stripe patterns on leaves and black spots on grapes.',
    7: 'Squash Powdery Mildew: A common fungal disease in squash, with white, powdery coating on leaves, potentially stunting plant growth.',
    8: 'Corn (Maize) Common Rust: A widespread fungal infection in corn characterized by rusty brown pustules on leaves, affecting photosynthesis and yield.',
    9: 'Tomato Early Blight: A fungal disease causing dark, concentric spots on tomato leaves, stems, and sometimes fruits, leading to premature leaf drop.',
    10: 'Tomato Yellow Leaf Curl Virus: A viral disease causing tomato leaves to curl and yellow, stunting the plant\'s growth and reducing yield.',
    11: 'Healthy Grape: Grapevines in prime condition, showing no signs of disease, with healthy leaves and developing fruit clusters.',
    12: 'Healthy Peach: Peach trees exhibiting vigorous growth, free from any diseases or abnormalities, with healthy leaves and fruit.',
    13: 'Healthy Blueberry: Blueberry bushes in optimal health, free from disease, with lush foliage and developing berries.',
    14: 'Apple Apple Scab: A fungal disease causing olive-green to black spots on apple leaves and fruits, potentially leading to fruit deformation.',
    15: 'Orange Huanglongbing (Citrus Greening): A severe bacterial disease in orange trees causing yellowing of leaves, misshapen fruits, and reduced yield.',
    16: 'Tomato Spider Mites Infestation: Two-spotted spider mites causing yellow speckling on leaves, leading to leaf drop and reduced plant vigor.',
    17: 'Potato Late Blight: A devastating fungal disease causing dark, greasy lesions on leaves and tubers, famously associated with the Irish potato famine.',
    18: 'Tomato Mosaic Virus: A viral disease causing mottled green and yellow leaves, stunted growth, and malformed fruits in tomato plants.',
    19: 'Healthy Apple: Apple trees exhibiting excellent health, with vibrant leaves and developing fruit, free from diseases.',
    20: 'Healthy Soybean: Soybean plants in optimal health, showing no signs of diseases or nutrient deficiencies, with healthy leaves and pods.',
    21: 'Peach Bacterial Spot: A bacterial disease causing small, dark spots on peach leaves and fruit, leading to fruit cracking and reduced yield.',
    22: 'Healthy Pepper, Bell: Bell pepper plants in excellent condition, showing vigorous growth and no signs of diseases or pest infestations.',
    23: 'Healthy Tomato: Tomato plants exhibiting robust health, free from diseases and pests, with healthy foliage and developing fruits.',
    24: 'Corn (Maize) Northern Leaf Blight: A fungal disease characterized by long, cigar-shaped gray-green lesions on corn leaves, affecting photosynthesis and yield.',
    25: 'Healthy Potato: Potato plants in prime health, showing vigorous growth, with no signs of diseases or nutrient deficiencies.',
    26: 'Tomato Late Blight: A serious fungal disease causing dark, water-soaked lesions on leaves and fruits, leading to rapid decay.',
    27: 'Tomato Bacterial Spot: A bacterial disease causing small, water-soaked lesions on leaves and fruits, leading to leaf and fruit spots.',
    28: 'Apple Black Rot: A fungal disease causing dark, sunken lesions on apple fruit and leaves, potentially leading to significant fruit loss.',
    29: 'Grape Black Rot: A fungal disease in grapes causing brownish-black lesions on fruit and leaves, leading to shriveled, inedible grapes.',
    30: 'Cherry Powdery Mildew: A fungal infection in cherry trees, including sour varieties, characterized by a white powdery growth on leaves and fruit.',
    31: 'Healthy Strawberry: Strawberry plants in excellent health, with vibrant green leaves and developing fruit, free from diseases.',
    32: 'Tomato Septoria Leaf Spot: A fungal disease causing small, circular spots with gray centers on tomato leaves, leading to leaf yellowing and drop.',
    33: 'Pepper, Bell Bacterial Spot: A bacterial disease causing small, water-soaked spots on leaves and fruits of bell pepper plants.',
    34: 'Corn (Maize) Cercospora Leaf Spot: A fungal disease causing grayish spots with red or purple halos on corn leaves, affecting growth and yield.',
    35: 'Potato Early Blight: A fungal disease causing irregular brown spots on leaves, leading to premature leaf wilting and reduced tuber size.',
    36: 'Grape Leaf Blight (Isariopsis Leaf Spot): A fungal disease in grapevines causing large, irregularly shaped brown spots on leaves.',
    37: 'Healthy Cherry: Cherry trees, including sour varieties, in optimal health, with lush foliage and developing cherries, free from diseases.'
}


class CNNModel(nn.Module):
    def __init__(self, num_classes=38):
        super(CNNModel, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Dropout layer
        self.dropout = nn.Dropout(0.7)
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 32 * 32)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)
        return x

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load your model
model = CNNModel()  # Replace with your model class
model.load_state_dict(torch.load('model_epoch_10.pth', map_location=torch.device('cpu')))

model.eval()

# Define a transform to preprocess the image
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Define a route for your API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        # Open the image and check its size
        image = Image.open(io.BytesIO(file.read()))

        # Check if the image size is 256x256
        if image.size != (256, 256):
            return jsonify({'error': 'Invalid image size. Image must be 256x256 pixels'}), 400

        # Convert the image to the format expected by your model
        image = transform(image).unsqueeze(0)

        # Make a prediction
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            prediction = predicted.item()  # Or process as needed for your model

        # Return the result
        prediction = index_to_label[prediction].split('___')
        if len(prediction) == 2:
            plant, disease = prediction
        else:
            return jsonify({'error': 'Invalid prediction format'}), 500
        
        return jsonify({'plant': plant, 'disease': disease, 'description': index_to_description[predicted.item()]})


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
