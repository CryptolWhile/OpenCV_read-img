import cv2
# 1. Mo Camera
cap = cv2.VideoCapture(0)

#load Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 2.1 Chuyen sang grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2.2 Phat hien canh
    edges = cv2.Canny(gray, 100, 200)

    # 2.3 Phat hien mat
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # 2.4 Hien thi ket qua 
    cv2.imshow('Original', frame)
    cv2.imshow('Edges', edges)

    # Thoat khi nhan phim 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Giai phong tai nguyen
cap.release()
cv2.destroyAllWindows()