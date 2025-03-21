from imagededup.methods import PHash
import os

phasher = PHash()

image_dir = 'filtered_output'

# Generate encodings for all images in an image directory
encodings = phasher.encode_images(image_dir=image_dir)

# Find duplicates using the generated encodings
duplicates = phasher.find_duplicates_to_remove(encoding_map=encodings, max_distance_threshold=4)

for x in duplicates:
    path = os.path.join(image_dir, x)
    os.remove(path)