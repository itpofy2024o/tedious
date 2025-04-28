import tensorflow as tf
import numpy as np
import sys
import os
from PIL import Image
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()

# Define image size (must match training script)
img_height, img_width = 224, 224

# Class labels (must match training script's class indices)
class_labels = ['dm', 'is', 'jas', 'kyw', 'mm', 'yua', 'yw']

def load_and_preprocess_image(image_path):
    """Load and preprocess a single image for prediction."""
    # Check if image exists and is a PNG
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image {image_path} does not exist.")
    if not image_path.lower().endswith('.png'):
        raise ValueError("Image must be a PNG file.")
    
    # Load image
    img = Image.open(image_path).convert('RGB')
    img = img.resize((img_width, img_height))
    img_array = np.array(img) / 255.0  # Rescale to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict_image(model, image_path):
    """Predict the class of an input image."""
    img_array = load_and_preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = class_labels[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx]
    return predicted_class, confidence

if __name__ == "__main__":
    # Check command-line arguments
    if len(sys.argv) != 2:
        print("Usage: python predict_portrait_mobilenet.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    try:
        # Load the trained model
        model_path = 'best_mobilenet_model.keras'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} does not exist. Ensure training completed successfully.")
        model = tf.keras.models.load_model(model_path)
        
        # Predict
        predicted_class, confidence = predict_image(model, image_path)
        print(f"Predicted Class: {predicted_class}")
        print(f"Confidence: {confidence:.4f}")
        
        # Display all class probabilities
        predictions = model.predict(load_and_preprocess_image(image_path))[0]
        print("\nAll Class Probabilities:")
        for label, prob in zip(class_labels, predictions):
            print(f"{label}: {prob:.4f}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)