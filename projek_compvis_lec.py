import cv2

# Cascade Classifier
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_mcs_mouth.xml')

# Attributes
threshold_value = 90
thickness = 2
font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN
org = (30, 30)
white_color = (255, 255, 255)
red_color = (0, 0, 255)
green_color = (0, 255, 0)
txt_no_face = "No FACE found!"
txt_no_mask = "Use Your MASK!"
txt_mask_found = "MASK detected!"

# Read video
video_cap = cv2.VideoCapture(0)

while True:

    # Get individual frame
    ret, frame = video_cap.read()

    # Flip frame horizontally
    img_flip = cv2.flip(frame, 1)

    # Convert frame to grayscale
    img_gray = cv2.cvtColor(img_flip, cv2.COLOR_BGR2GRAY)

    # Convert frame to black and white
    thresh, img_bw = cv2.threshold(img_gray, threshold_value, 255, cv2.THRESH_BINARY)

    # Detect faces on frame grayscale
    faces_gray = face_cascade.detectMultiScale(img_gray, scaleFactor = 1.2, minNeighbors = 3)

    # Detect faces on frame black and white
    faces_bw = face_cascade.detectMultiScale(img_bw, scaleFactor = 1.2, minNeighbors = 3)

    if (len(faces_gray) == 0 and len(faces_bw) == 0):

        cv2.putText(img_flip, txt_no_face, org, font, font_scale, red_color, thickness)

    elif (len(faces_gray) == 0 and len(faces_bw) == 1):

        cv2.putText(img_flip, txt_mask_found, org, font, font_scale, green_color, thickness)

    else:

        # Draw rectangle on detected faces
        for face_rectangle in faces_gray:

            x, y, height, width = face_rectangle

            cv2.rectangle(img_flip, (x, y), (x + width, y + height), white_color, 2)

            cropped_face_gray = img_gray[y:y + height, x:x + width]
            cropped_face_bgr = img_flip[y:y + height, x:x + width]

            # Detect lips on face
            detected_mouth = mouth_cascade.detectMultiScale(img_gray, scaleFactor = 1.2, minNeighbors = 3)

            if (len(detected_mouth) == 0):
                
                cv2.putText(img_flip, txt_mask_found, org, font, font_scale, green_color, thickness)

            else:
                
                for mouth_rectangle in detected_mouth:

                    mouth_x, mouth_y, mouth_height, mouth_width = mouth_rectangle

                    if (y < mouth_y < y + height):

                        cv2.putText(img_flip, txt_no_mask, org, font, font_scale, red_color, thickness)

                        cv2.rectangle(img_flip, (mouth_x, mouth_y), (mouth_x + mouth_width, mouth_y + mouth_height), red_color, 3)

                        break


    cv2.imshow("Mask Detection", img_flip)

    # Press 'q' on keyboard to exit video window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_cap.release()
cv2.destroyAllWindows

