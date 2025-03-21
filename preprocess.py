import os
import cv2
import shutil
from PIL import Image

# 1. Define source and destination folders
source_folder = "extracted_faces_ssd"
destination_folder = "filtered_output"

# 2. Define thresholds
# area_threshold = 3000     # Example: minimum area (width * height)
blur_threshold = 50    # Example: variance of Laplacian below this = "blurry"

# 3. Create destination folder if it doesn't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# 4. Function to detect blur using variance of Laplacian
def is_blurry(image_path, threshold=100.0):
    """
    Returns True if the image is blurry.
    The lower the variance, the blurrier the image.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        # Could not read the image in OpenCV; consider it invalid
        return True
    
    laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
    return laplacian_var < threshold

# 5. Loop over images in source folder
for file_name in os.listdir(source_folder):
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        source_path = os.path.join(source_folder, file_name)
        
        try:
            # Check area with PIL
            with Image.open(source_path) as img:
                w, h = img.size
                area = w * h

            # Check if area is above threshold
            #if area > area_threshold:
                # Check blur
            if not is_blurry(source_path, blur_threshold):
                # If not blurry and area is large enough, copy the file
                dest_path = os.path.join(destination_folder, file_name)
                shutil.copy2(source_path, dest_path)
                print(f"Copied {file_name} to {destination_folder}")
            else:
                print(f"Skipped {file_name} (blurry)")
            # else:
            #     print(f"Skipped {file_name} (area too small: {area})")

        except Exception as e:
            print(f"Error processing {file_name}: {e}")
