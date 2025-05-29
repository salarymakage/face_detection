# Detection code

import cv2
import numpy as np
import sqlite3

def getprofile(id):
    """Retrieve user profile by ID from the database."""
    try:
        conn = sqlite3.connect("sqlite.db")
        cursor = conn.execute("SELECT * FROM STUDENTS WHERE id=?", (id,))
        profile = None
        for row in cursor:
            profile = row
        return profile
    finally:
        conn.close()

# Load the DNN face detector model 
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')
cam = cv2.VideoCapture(0)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')

def detect_faces_dnn(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            faces.append((x, y, x1 - x, y1 - y))
    return faces

try:
    while True:
        ret, img = cam.read()
        if not ret:
            print("Failed to capture frame")
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detect_faces_dnn(img)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, conf = recognizer.predict(gray[y:y + h, x:x + w])
            
            if conf < 100:  # You can adjust the threshold value based on your requirement
                profile = getprofile(id)
                if profile:
                    text = f"ID: {id} - Name: {profile[1]} - Age: {profile[2]}"
                else:
                    text = "Unknown"
            else:
                text = "Unknown"
            
            cv2.putText(img, text, (x, y + h + 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("FACE", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cam.release()
    cv2.destroyAllWindows()
