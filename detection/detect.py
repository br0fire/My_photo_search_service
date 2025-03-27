import os
import cv2
import argparse
from deepface import DeepFace
from tqdm import tqdm

def extract_faces(dataset_folder, output_folder, detector_backend="yolov8"):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through each image file in the dataset folder
    for filename in tqdm(os.listdir(dataset_folder), desc="Extracting faces"):
        file_path = os.path.join(dataset_folder, filename)

        # Process only files with common image extensions
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                # Use DeepFace to extract faces
                faces = DeepFace.extract_faces(
                    img_path=file_path,
                    detector_backend=detector_backend,
                    normalize_face=False,
                    color_face='bgr',
                    align=False
                )

                if faces:
                    base_filename, _ = os.path.splitext(filename)
                    for i, face_data in enumerate(faces):
                        face_img = face_data["face"]
                        output_filename = f"{base_filename}_face_{i+1}.jpg"
                        output_path = os.path.join(output_folder, output_filename)
                        cv2.imwrite(output_path, face_img)
                else:
                    print(f"No faces found in {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Extract faces from images using DeepFace.")
    parser.add_argument("--input", "-i", required=True, help="Path to input folder containing images.")
    parser.add_argument("--output", "-o", required=True, help="Path to output folder for extracted faces.")
    parser.add_argument("--backend", "-b", default="yolov8", help="Face detector backend (default: yolov8).")

    args = parser.parse_args()
    extract_faces(args.input, args.output, args.backend)

if __name__ == "__main__":
    main()
