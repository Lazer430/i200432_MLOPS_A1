from flask import Flask, render_template, request
# import joblib
import numpy as np
from PIL import Image
import io
import pickle

app = Flask(__name__)

# Load your pre-trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define the allowed file extensions for image uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    
    # If user does not select file, browser also submit an empty part without filename
    if file.filename == '':
        return "No selected file"
    
    # If the file has an allowed extension
    if file and allowed_file(file.filename):
        # model = joblib.load('model.pkl')
        with open('model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        image = Image.open(io.BytesIO(file.read()))
        # image = plt.imread(image_path)
        image = np.array(image)
        image = image.reshape(-1, 28, 28, 1)
        prediction = model.predict(image)
        predicted_label = chr(np.argmax(prediction) + 65)
        return render_template('index.html', label=predicted_label)

    else:
        return "Invalid file format"

if __name__ == '__main__':
    app.run(debug=True)
