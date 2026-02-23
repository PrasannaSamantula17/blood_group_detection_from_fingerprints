from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS

# Define the model path
model_path = os.path.join("model", "model_best.h5")

# Load the model
try:
    print("Current directory:", os.getcwd())
    print("Files available:", os.listdir())
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print("Error loading model:", str(e))
    model = None

# Define allowed file extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(file):
    """
    Preprocesses the image exactly as in predict_blood_group.
    """
    # Save the uploaded file temporarily to load with load_img
    temp_path = "temp_image." + file.filename.rsplit(".", 1)[1].lower()
    file.save(temp_path)
    
    # Load and preprocess the image
    img = load_img(temp_path, target_size=(256, 256))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Clean up temporary file
    os.remove(temp_path)
    
    print("Image array shape:", img_array.shape)  # Should be (1, 256, 256, 3)
    return img_array

@app.route('/')
def home():
    return render_template("indexed.html")

@app.route("/predict", methods=["POST"])
def predict():
    print("Received request at /predict")

    if model is None:
        print("Model not loaded")
        return jsonify({"error": "Model not loaded"}), 500

    if "file" not in request.files:
        print("No file provided")
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    print(f"Received file: {file.filename}")

    if file.filename == "":
        print("No file selected")
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        print("Invalid file type")
        return jsonify({"error": "Invalid file type. Allowed types are png, jpg, jpeg, bmp"}), 400

    try:
        # Preprocess the image
        img_array = preprocess_image(file)
        print("Image preprocessed successfully!")
    except Exception as e:
        print(f"Image preprocessing failed: {str(e)}")
        return jsonify({"error": f"Image preprocessing failed: {str(e)}"}), 400

    try:
        # Perform prediction
        prediction = model.predict(img_array)
        print("Raw prediction output:", prediction[0])  # Log raw output for debugging
        predicted_class = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0]))
        print(f"Predicted class index: {predicted_class}, Confidence: {confidence}")
    except Exception as e:
        print(f"Model prediction failed: {str(e)}")
        return jsonify({"error": f"Model prediction failed: {str(e)}"}), 500

    # Define class names (exact match with predict_blood_group)
    class_names = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']
    if predicted_class >= len(class_names):
        return jsonify({"error": "Predicted class index out of range"}), 500
    predicted_label = class_names[predicted_class]

    # Return the result as JSON
    return jsonify({
        "predicted_class": predicted_class,
        "predicted_label": predicted_label,
        "confidence": confidence
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
