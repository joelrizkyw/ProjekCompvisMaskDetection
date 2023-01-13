import numpy as np
import cv2
import os

# Read train image
train_image_path = "assets/train/"

# Cascade
face_cascade = cv2.CascadeClassifier("./haarcascade/haarcascade_frontalface_default.xml")

# Nama nama label
mask_label_list = os.listdir(train_image_path)

for idx, mask_label in enumerate(mask_label_list):

    # Path mask label
    image_path = train_image_path + mask_label

    for image in os.listdir(image_path):

        # Full path image
        image_full_path = image_path + "/" + image

        # Adjust threshold value in range 80 to 105 based on your light.
        bw_threshold = 80

        # Baca image dari setiap full path image
        
        img = cv2.imread(image_full_path, 0)

        # Convert image ke gray juga
        img_gray = cv2.

        # Convert image in black and white
        (thresh, img_bw) = cv2.threshold(img_gray, bw_threshold, 255, cv2.THRESH_BINARY)

        # Detect faces gray
        detected_faces_gray = face_cascade.detectMultiScale(img_gray, 1.2, 5)

        # Detect faces black and white
        detected_faces_bw = face_cascade.detectMultiScale(img_bw, 1.2, 5)

        if (len(detected_faces_gray) == 0 and len(detected_faces_bw) == 0):

            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 1)


