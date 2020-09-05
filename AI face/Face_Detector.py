import cv2
from random import randrange

# load dat form frontal cv
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# video image read funtion from
webcam = cv2.VideoCapture(0)


while True:
    successful_frame_read, frame = webcam.read()
    # Must convert to grayscale
    garyscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_face_data.detectMultiScale(garyscaled_img)

    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    # cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 3)

    cv2.imshow('janith is the best', frame)
    key = cv2.waitKey(2)

    if key == 81 or key == 113:
        break
webcam.release()

# cv2.imshow('RDJ image', img)
# cv2.waitKey()
#
# garyscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# face_coordinates = trained_face_data.detectMultiScale(garyscaled_img)
#
# print(face_coordinates)
#
# # cv2.rectangle(img, (700, 600), (162,162), (0, 255, 0), 2)
# for (x, y, w, h) in face_coordinates:
#     cv2.rectangle(img, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 3)
#
# # display the image
# cv2.imshow('janith is the best', img)
# cv2.waitKey()
#
# print("code completed")
