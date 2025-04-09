from PIL import Image
import os
import sys

aaa = sys.argv[1]

def get_image_dimensions(image_path):
    """
    Read and return the dimensions (width, height) of a single image.
    
    Args:
        image_path (str): Path to the image file.
    
    Returns:
        tuple: (width, height) in pixels, or None if the image cannot be read.
    """
    try:
        # Open the image file
        with Image.open(image_path) as img:
            # Get dimensions
            width, height = img.size
            return width, height
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return None

# Example usage
if __name__ == "__main__":
    # Specify the path to your image
    image_path = aaa  # Replace with your image path
    
    # Check if the file exists
    if os.path.exists(image_path):
        dimensions = get_image_dimensions(image_path)
        if dimensions:
            width, height = dimensions
            print(f"Image dimensions: {width} x {height} pixels")
    else:
        print(f"File {image_path} does not exist.")
