from flask import Flask, request, jsonify
import waitress
import model as model
from PIL import Image
import os

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:
        img = Image.open(file).convert("RGB")
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
    # Save the image to a temporary path
    temp_image_path = "/tmp/temp_image.jpg"
    img.save(temp_image_path)
    
    # Run the prediction
    predicted_class, confidence = model.load_and_predict(temp_image_path)
    
    # Define class labels
    class_labels = ["Alkaline", "Li-ion", "Lithium", "Ni-CD", "Ni-MH"]
    
    return jsonify({
        "predicted_class": class_labels[predicted_class],
        "confidence": confidence
    })

if __name__ == "__main__":
    print("Server starting...")
    waitress.serve(app, host='0.0.0.0', port=5000)