import numpy as np
import cv2
import face_recognition
import pickle
import tensorflow as tf


def SVM_smile(npimage: np.ndarray):
    with open("D:\\svm_model.sav", "rb") as content:
        model = pickle.load(content)
        return model.predict(npimage)


def SqNN_smile(npimage: np.ndarray):
    model = tf.keras.models.load_model("./../../NMKHMT/model")
    model.predict(npimage)


def capturing_from_webcam(algorithm):
    #the video capturer:
    cap = cv2.VideoCapture(0)


    IMG_SIZE = 50
    #face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    

    while True:
        #get image from the captured video:
        ret, frame = cap.read()


        #gray scale image to make things eassier:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        

        #resized_gray = cv2.resize(gray, (0,0), fx=0.25, fy=0.25)
        """ faces = face_cascade.detectMultiScale(gray, 1.05, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5) """


        #rgb_frame = frame[:, :, ::-1]
        #get the location of human face in the grayscaled image:
        face_locations = face_recognition.face_locations(gray)


        #smile detection process:
        for (top, right, bottom, left) in face_locations:
            #smile detection:
            face_gray = gray[top:bottom, left:right] #get only the face

            #preprocessing image:
            face_gray = cv2.resize(face_gray, (IMG_SIZE, IMG_SIZE))
            test_face_gray = np.array(face_gray)
            test_face_gray = test_face_gray/255.0
            test_face_gray = test_face_gray.flatten().reshape(-1, IMG_SIZE*IMG_SIZE)

            isSmiling, state = 0, "Smiling" #some needed variables

            #choose the algorithm to make a detection:
            if (algorithm == "SVM"): isSmiling = SVM_smile(test_face_gray)
            elif (algorithm == "SqNN"): isSmiling = SqNN_smile(test_face_gray)

            if (isSmiling == 0): state = "Not smiling"

            #draw the rectangle for each detected human face detected:
            cv2.rectangle(frame, (left, top), (right, bottom), (225, 0, 0), 3)

            #draw the text "Smiling"/"Not smiling" with box:
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (225, 0, 0), cv2.FILLED)
            cv2.putText(frame, state, (left + 6, bottom - 6), cv2.FONT_HERSHEY_COMPLEX, 1.0, 
                        (225, 225, 225), 1)

        #showing to screen
        cv2.imshow('Captured face', frame)

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    capturing_from_webcam("SVM")
