from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import base64
import os
app = Flask(__name__)


model_path = r"C:\Users\HP\Desktop\imageReconizer\backend\model\fruit_classifier_model.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Request received")

        # Parse the JSON data
        data = request.get_json()
        if data is None:
            print("No data received")
            return jsonify({'error': 'No data received'}), 400

        if 'image' not in data:
            print("No image data received")
            return jsonify({'error': 'No image data received'}), 400

        # Decode the Base64 image
        image_data = data['image']
        try:
            image_bytes = base64.b64decode(image_data)
            print(f"Decoded image bytes: {len(image_bytes)} bytes")
            print(f"First 20 bytes: {image_bytes[:20]}")

            # Save the image to a file for manual inspection
            with open("received_image.jpg", "wb") as f:
                f.write(image_bytes)

            # Use PIL to read the image
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        except Exception as e:
            print(f"Error decoding image: {e}")
            return jsonify({'error': f"Error decoding image: {e}"}), 400

        # Resize image to match the input size of your object detection model
        img = img.resize((224, 224))  # Updated resolution
        img_array = np.array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Convert to TensorFlow tensor
        input_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

        # Run inference with the model
        detections = model(input_tensor)

        # Extract bounding boxes, classes, and scores
        boxes = detections['detection_boxes'][0].numpy()  # Bounding boxes
        classes = detections['detection_classes'][0].numpy().astype(int)  # Class IDs
        scores = detections['detection_scores'][0].numpy()  # Confidence scores

        # Confidence threshold for filtering
        confidence_threshold = 0.5
        results = []

        for i in range(len(scores)):
            if scores[i] >= confidence_threshold:
                box = boxes[i].tolist()  # [ymin, xmin, ymax, xmax]
                label = classes[i]  # Use the class ID directly
                confidence = scores[i]
                results.append({
                    'label': label,  # You can map this to a string if the model returns class names
                    'confidence': float(confidence),
                    'box': box
                })

        # Return predictions to the client
        return jsonify({'predictions': results})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)