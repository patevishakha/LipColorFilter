import cv2
import numpy as np
import dlib

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        #print(face)
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 4)

        face_landmarks = predictor(gray,face)
        points_list = []  #contains all the landmark points
        for i in range(0,68):
            x = face_landmarks.part(i).x
            y = face_landmarks.part(i).y
            points_list.append([x,y])
            cv2.circle(frame,(x,y),2,(255,0,255),-1)
            # cv2.putText(frame,str(i),(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.2,(0,255,255),1)  #for lankmark number
        print(points_list)
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break