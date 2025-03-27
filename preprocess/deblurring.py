import os
import cv2
import shutil
import argparse

# Function to detect blur using variance of Laplacian
def is_blurry(image_path, threshold=100.0):
    """
    Returns True if the image is blurry.
    The lower the variance, the blurrier the image.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return True
    laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
    return laplacian_var < threshold

def filter_images(source_folder, destination_folder, blur_threshold):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for file_name in os.listdir(source_folder):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            source_path = os.path.join(source_folder, file_name)

            try:
                if not is_blurry(source_path, blur_threshold):
                    dest_path = os.path.join(destination_folder, file_name)
                    shutil.copy2(source_path, dest_path)
                else:
                    print(f"Skipped {file_name} (blurry)")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Filter out blurry images.")
    parser.add_argument("--input", "-i", required=True, help="Path to the folder with input images.")
    parser.add_argument("--output", "-o", required=True, help="Path to save filtered (non-blurry) images.")
    parser.add_argument("--blur-threshold", "-t", type=float, default=100.0, help="Blur detection threshold (default: 100.0).")

    args = parser.parse_args()
    filter_images(args.input, args.output, args.blur_threshold)

if __name__ == "__main__":
    main()
