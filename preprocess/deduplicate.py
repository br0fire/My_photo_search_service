import os
import argparse
from imagededup.methods import PHash

def remove_duplicates(image_dir, max_distance_threshold=4):
    phasher = PHash()

    # Generate encodings for all images in an image directory
    encodings = phasher.encode_images(image_dir=image_dir)

    # Find duplicates using the generated encodings
    duplicates = phasher.find_duplicates_to_remove(encoding_map=encodings, max_distance_threshold=max_distance_threshold)

    # Remove duplicate files
    for file_name in duplicates:
        path = os.path.join(image_dir, file_name)
        try:
            os.remove(path)
            print(f"Removed duplicate: {file_name}")
        except Exception as e:
            print(f"Error removing {file_name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Remove visually duplicate images using perceptual hashing (PHash).")
    parser.add_argument("--input", "-i", required=True, help="Path to the folder with images.")
    parser.add_argument("--threshold", "-t", type=int, default=4, help="Max Hamming distance to consider images as duplicates (default: 4).")

    args = parser.parse_args()
    remove_duplicates(args.input, args.threshold)

if __name__ == "__main__":
    main()
