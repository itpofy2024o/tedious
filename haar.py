import cv2
import sys
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
img = cv2.imread(sys.argv[1])
print(img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
if len(faces) > 0:
    x, y, w, h = faces[0]
    face = img[y:y+h, x:x+w]
    face = cv2.resize(face, (224, 224))
    cv2.imwrite('{}.png'.format(sys.argv[2]), face)
else:
    print("No face detected in test image.")