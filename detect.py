from ultralytics import YOLO
import gdown
import os

WEIGHT_URL = "https://drive.google.com/uc?id=1qcr9DbgsX3ryrz2uU8w4Xm3cOrRywXqb"
PATH = "yolov8n-face.pt"
if __name__ == "__main__":
    if not os.path.isfile(PATH):
        gdown.download(WEIGHT_URL, PATH, quiet=False)
    yolo = YOLO(PATH)
    img_path = 'img1.jpg'
    result = yolo(img_path, verbose=False, show=False, conf=0.25)[0]
    print(result)
    # image = Image.fromarray((face_objs[0]['face'] * 255).astype(np.uint8))
    # image.save('detected_image.jpg')
