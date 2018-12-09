import numpy as np
import cv2 as cv

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')
cap = cv.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        rectregion_gray = gray[y:y+h, x:x+w]
        rectregion_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(rectregion_gray)
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(rectregion_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    cv.imshow('img', img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()