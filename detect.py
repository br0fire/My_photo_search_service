import os
import cv2
from deepface import DeepFace
from tqdm import tqdm

# Define input and output folders
dataset_folder = "dataset1"
output_folder = "extracted_faces_ssd"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through each image file in the dataset folder
for filename in tqdm(os.listdir(dataset_folder)):
    file_path = os.path.join(dataset_folder, filename)
    
    # Process only files with common image extensions
    if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        #print(f"Processing {filename}...")
        try:
            # Use DeepFace to extract faces; enforce_detection=False prevents errors when no face is found
            faces = DeepFace.extract_faces(img_path=file_path, detector_backend="ssd", normalize_face=False, color_face='bgr', align=False)
            
            if faces:
                # Use the base filename for saving faces
                base_filename, _ = os.path.splitext(filename)
                for i, face_data in enumerate(faces):
                    face_img = face_data["face"]
                    # Save each face with a unique name
                    output_filename = f"{base_filename}_face_{i+1}.jpg"
                    output_path = os.path.join(output_folder, output_filename)
                    cv2.imwrite(output_path, face_img)
                    #print(f"Saved {output_filename}")
            else:
                print(f"No faces found in {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
