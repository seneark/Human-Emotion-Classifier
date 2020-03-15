from keras.models import load_model
from keras.preprocessing.image import img_to_array, image
import cv2
import numpy as np

Haar_Face = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
classifier = load_model('./Emotion_model.h5')

Emotions = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    labels = []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = Haar_Face.detectMultiScale(gray, 1.3, 6)

    for (x, y, w, h) in faces:
        cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            pred = classifier.predict(roi)[0]
            label = Emotions[pred.argmax()]
            label_position = (x, y)
            cv2.putText(gray, label, label_position,
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
        else:
            pass

    cv2.imshow('Emotion Detector', gray)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
