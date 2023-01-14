import numpy as np
import cv2
import os

# Read train image
train_image_path = "assets/train/"

# Cascade
face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
mouth_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_mcs_mouth.xml")

faces_image_gray_list = []
class_id_list = []

# Nama nama label
mask_label_list = os.listdir(train_image_path)

for idx, mask_label in enumerate(mask_label_list):

    # Path mask label
    image_path = train_image_path + mask_label

    print(mask_label)

    for image in os.listdir(image_path):

        # Full path image
        image_full_path = image_path + "/" + image

        print(image_full_path)

        # Adjust threshold value in range 80 to 105 based on your light.
        bw_threshold = 80

        # Baca image dari setiap full path image
        img = cv2.imread(image_full_path)

        # Resize image
        scale_percent = 50 # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)

        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

        # Convert image ke gray juga
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Convert image in black and white
        (thresh, img_bw) = cv2.threshold(img_gray, bw_threshold, 255, cv2.THRESH_BINARY)

        # Detect faces gray
        detected_faces_gray = face_cascade.detectMultiScale(img_gray, scaleFactor = 1.2, minNeighbors = 3)


        # Detect faces black and white
        detected_faces_bw = face_cascade.detectMultiScale(img_bw, scaleFactor = 1.2, minNeighbors = 5)

        
        print("detected faces gray: ", len(detected_faces_gray))

        # if (len(detected_faces_gray) == 0 and len(detected_faces_bw) == 0):
            
        #     print("No mask found")

        # elif (len(detected_faces_gray) == 0 and len(detected_faces_bw) == 1):

        #     print("Mask detected")

        # else:

        #     for face_rectangle in detected_faces_gray:

        #         x, y, h, w = face_rectangle

        #         # Melakukan crop image dengan panjang (y:y+h) dan lebar (x:x+w)
        #         faces_image = img[y:y + h, x:x + w]
        #         faces_image_gray = img_gray[y:y + h, x:x + w]

        #         faces_image_gray_list.append(faces_image_gray)
        #         class_id_list.append(idx)

                
# face_recognizer = cv2.face.LBPHFaceRecognizer_create()
# face_recognizer.train(faces_image_gray_list, np.array(class_id_list))
                
# Read test image
# test_image_path = "assets/test/"

# for test_image in os.listdir(test_image_path):

#     # Full path untuk setiap test image
#     test_image_full_path = test_image_path + test_image

#     # Baca test image dengan full path image test-nya
#     img = cv2.imread(test_image_full_path)

#     # Convert test image menjadi grayscale
#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Convert image in black and white
#     (thresh, img_bw) = cv2.threshold(img_gray, bw_threshold, 255, cv2.THRESH_BINARY)

#     # Detect faces gray
#     detected_faces_gray = face_cascade.detectMultiScale(img_gray, 1.2, 5)

#     # Detect faces black and white
#     detected_faces_bw = face_cascade.detectMultiScale(img_bw, 1.2, 5)
    
#     if (len(detected_faces_gray) == 0 and len(detected_faces_bw) == 0):

#         cv2.putText(img, "No face found...", (30, 30), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 1)

#     elif (len(detected_faces_gray) == 0 and len(detected_faces_bw) == 1):
        
#         cv2.putText(img, "Mask detected", (30, 30), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 1)

#     else:

#         for face_rectangle in detected_faces_gray:

#                 x, y, h, w = face_rectangle

#                 # Melakukan crop image dengan panjang (y:y+h) dan lebar (x:x+w)
#                 faces_image = img[y:y + h, x:x + w]
#                 faces_image_gray = img_gray[y:y + h, x:x + w]

#                 # confidence semakin kecil semakin bagus result nya
#                 result, confidence = face_recognizer.predict(faces_image_gray)

#                 cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)

#                 # Deteksi bibir
#                 detected_mouths = mouth_cascade.detectMultiScale(img_gray, 1.5, 5)

#                 if (len(detected_mouths) == 0):

#                     cv2.putText(img, "Mask detected", (30, 30), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 1)

#                 else:

#                     for mouth_rectangle in detected_mouths:

#                         mx, my, mh, mw = mouth_rectangle

#                         if (y < my < y + h):

#                             cv2.putText(img, "Mask detected", (30, 30), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 1)

#                             # cv2.imshow("Result", img)
#                             # cv2.waitKey(0)

