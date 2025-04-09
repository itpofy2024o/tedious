from PIL import Image
import os
import sys

aaa = sys.argv[1]

def get_folder_image_dimensions(folder_path):
    total_width = 0
    total_height = 0
    valid_image_count = 0
    
    # Supported image extensions
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    extra = []
    try:
        # Iterate through all files in the folder
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(image_extensions):
                image_path = os.path.join(folder_path, filename)
                with Image.open(image_path) as img:
                    width, height = img.size
                    if str(width).endswith("0")!=True or str(height).endswith("0")!=True:
                        print(str(width).endswith("0")==True,width,str(height).endswith("0")==True,height)
                        extra.append([filename,width,height])
                    total_width += width
                    total_height += height
                    valid_image_count += 1
                    print(f"Processed {filename}: {width} x {height} pixels")
        
        if valid_image_count == 0:
            print(f"No valid image files found in {folder_path}")
            return None
        else:
            print(f"\nTotal number of images processed: {valid_image_count}")
            print(f"Total width: {total_width} pixels")
            print(f"Total height: {total_height} pixels")
            return total_width, total_height, valid_image_count, extra
    
    except Exception as e:
        print(f"Error processing folder {folder_path}: {e}")
        return None

# Example usage
if __name__ == "__main__":
    # Specify the path to your folder
    folder_path = aaa  # Replace with your folder path
    
    # Check if the folder exists
    if os.path.exists(folder_path):
        dimensions = get_folder_image_dimensions(folder_path)
        if dimensions:
            total_width, total_height, valid_image_count, ex = dimensions
            ex.sort(key=lambda x: x[0])
            avg_width = total_width / valid_image_count
            avg_height = total_height / valid_image_count
            print(f"Average width: {avg_width:.2f} pixels")
            print(f"Average height: {avg_height:.2f} pixels")
            print(list(ex),len(ex))
    else:
        print(f"Folder {folder_path} does not exist.")
