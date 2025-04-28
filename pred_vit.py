import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import sys
import os

# Load OpenPose model (download from https://github.com/CMU-Perceptual-Computing-Lab/openpose)
proto_file = "pose_deploy.prototxt"  # Path to OpenPose prototxt
weights_file = "pose_iter_440000.caffemodel"  # Path to OpenPose weights
net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)

# Load trained ViT model
model_path = "vit_classifier.keras"  # Update with your trained model path
classification_model = load_model(model_path)

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess variable-size PNG using OpenPose to crop around body and resize."""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        return None, "Error: Could not load image."

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Pose estimation using OpenPose
    height, width = image.shape[:2]
    inp = cv2.dnn.blobFromImage(image, 1.0 / 255, (width, height), (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inp)
    output = net.forward()
    
    # Extract key points (e.g., nose, neck, shoulders)
    points = []
    for i in range(18):  # OpenPose detects 18 key points
        prob_map = output[0, i, :, :]
        min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)
        x = (width * point[0]) / output.shape[3]
        y = (height * point[1]) / output.shape[2]
        if prob > 0.1:  # Threshold for valid key points
            points.append((int(x), int(y)))
        else:
            points.append(None)
    
    # If no key points detected, return original resized image
    if not any(points):
        image = cv2.resize(image, target_size)
        image = img_to_array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        return image, "Warning: No human detected, using resized image."

    # Find bounding box around detected body
    valid_points = [p for p in points if p]
    if not valid_points:
        image = cv2.resize(image, target_size)
        image = img_to_array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        return image, "Warning: No valid key points, using resized image."

    min_x = max(0, min(x for x, y in valid_points))
    max_x = min(width, max(x for x, y in valid_points))
    min_y = max(0, min(y for x, y in valid_points))
    max_y = min(height, max(y for x, y in valid_points))
    
    # Add padding to ensure entire body is included
    padding = 20
    min_x = max(0, min_x - padding)
    min_y = max(0, min_y - padding)
    max_x = min(width, max_x + padding)
    max_y = min(height, max_y + padding)
    
    # Crop image to bounding box
    cropped_image = image[min_y:max_y, min_x:max_x]
    
    # Resize to target size
    resized_image = cv2.resize(cropped_image, target_size)
    
    # Convert to array and normalize
    resized_image = img_to_array(resized_image) / 255.0
    resized_image = np.expand_dims(resized_image, axis=0)
    
    return resized_image, None

def predict(image_path):
    """Predict class for a variable-size PNG using the trained ViT model."""
    # Preprocess image
    preprocessed_image, warning = preprocess_image(image_path)
    
    if preprocessed_image is None:
        return warning
    
    if warning:
        print(warning)
    
    # Predict using ViT model
    predictions = classification_model.predict(preprocessed_image, verbose=0)
    class_idx = np.argmax(predictions[0])
    class_labels = {0: 'dm', 1: 'is', 2: 'jas', 3: 'kyw', 4: 'mm', 5: 'yua', 6: 'yw'}
    class_label = class_labels[class_idx]
    confidence = predictions[0][class_idx]
    
    return f"Predicted class: {class_label}, Confidence: {confidence:.4f}"

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict_portrait_vit.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: Image path {image_path} does not exist.")
        sys.exit(1)
    
    result = predict(image_path)
    print(result)